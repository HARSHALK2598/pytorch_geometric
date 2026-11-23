# The structural optimizer of "What Makes a Desired Graph for Relational Deep
# Learning?" (https://arxiv.org/abs/2606.08491) on RelBench's rel-f1
# driver-top3 task: table gates, KNN and two-hop injection with a switch per
# relation, temporal-continuity edges and the variational bottleneck.
#
# Requires `pip install relbench pyg-lib` (temporal neighbor sampling).
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from relbench import load_dataset
from sklearn.metrics import roc_auc_score

from torch_geometric.contrib.nn.models import (
    KNNEdges,
    StructuralOptimizer,
    TwoHopEdges,
)
from torch_geometric.contrib.transforms import TemporalContinuity
from torch_geometric.contrib.utils import from_relbench
from torch_geometric.loader import NeighborLoader

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--channels', type=int, default=64)
parser.add_argument('--lambda_table', type=float, default=1e-3)
parser.add_argument('--lambda_template', type=float, default=1e-3)
parser.add_argument('--beta_vib', type=float, default=1e-3)
args = parser.parse_args()

dataset = load_dataset('rel-f1')
task = dataset.load_task('driver-top3')
# 1. The database graph, plus temporal-continuity relations (step B):
data = TemporalContinuity()(from_relbench(dataset.get_db()))
target = task.entity_table


def make_loader(split, shuffle):
    table = task.get_table(split, mask_input_cols=False)
    df = table.df
    seed_time = df[table.time_col].values.astype('datetime64[s]').astype(
        np.int64)
    loader = NeighborLoader(
        data, num_neighbors=[64, 64], time_attr='time',
        input_nodes=(target, torch.from_numpy(df[task.entity_col].values)),
        input_time=torch.from_numpy(seed_time), batch_size=512,
        shuffle=shuffle)
    return loader, torch.from_numpy(df[task.target_col].values).float()


loaders = {
    split: make_loader(split, split == 'train')
    for split in ['train', 'val']
}

# 2. A simple encoder: standardised numeric columns, one linear map per table
#    (a constant feature for tables without columns).
stats = {
    nt: (torch.nanmean(data[nt].x.float(), 0),
         torch.from_numpy(np.nanstd(data[nt].x.float().numpy(), 0)))
    for nt in data.node_types if 'x' in data[nt]
}


def features(store, node_type):
    if 'x' not in store:
        return torch.ones(store.num_nodes, 1)
    mean, std = stats[node_type]
    return torch.nan_to_num((store.x.float() - mean) / (std + 1e-6))


encoders = torch.nn.ModuleDict({
    nt: torch.nn.Linear(features(data[nt], nt).size(1), args.channels)
    for nt in data.node_types
})

# 3. The structural optimizer, every step on:
schema = [et for et in data.edge_types if not et[1].startswith('next__')]
temporal = [et for et in data.edge_types if et[1].startswith('next__')]
model = StructuralOptimizer(
    data.node_types, schema, args.channels, out_channels=1,
    gated_tables=[nt for nt in data.node_types if nt != target],
    templates=[KNNEdges(target, k=5), *TwoHopEdges.instances(schema)],
    injected_edge_types=temporal, template_gates=True, vib=True,
    lambda_table=args.lambda_table, lambda_template=args.lambda_template,
    beta_vib=args.beta_vib)
optimizer = torch.optim.Adam([*model.parameters(), *encoders.parameters()],
                             lr=1e-2)


def run(split):
    loader, labels = loaders[split]
    training = split == 'train'
    model.train(training)
    encoders.train(training)
    preds, ys, total = [], [], 0.0
    for batch in loader:
        with torch.set_grad_enabled(training):
            x_dict = {
                nt: encoders[nt](features(batch[nt], nt))
                for nt in batch.node_types
            }
            n = batch[target].batch_size
            pred = model(x_dict, batch.edge_index_dict, target,
                         batch.batch_dict, batch[target].seed_time, n)
            y = labels[batch[target].input_id]
            loss = F.binary_cross_entropy_with_logits(pred.view(-1), y)
            if training:
                optimizer.zero_grad()
                (loss + model.structural_loss()).backward()
                optimizer.step()
        total += loss.item() * n
        preds.append(pred.view(-1).sigmoid().detach())
        ys.append(y)
    y, pred = torch.cat(ys), torch.cat(preds)
    return total / y.numel(), roc_auc_score(y.numpy(), pred.numpy())


for epoch in range(1, args.epochs + 1):
    train_loss, _ = run('train')
    _, val_auc = run('val')
    print(f'Epoch: {epoch:02d}, Loss: {train_loss:.4f}, '
          f'Val ROC-AUC: {val_auc:.4f}')

# Which tables and injected relations the model kept (evaluation mode):
model.eval()
tables = model.filter.table_gates
kept = [nt for nt in tables if tables[nt]().item() > 0]
print(f'Open tables ({len(kept)}/{len(tables)}): {kept}')
print(f'Open injected relations: {len(model.convs[0].open_edge_types())}'
      f'/{model.gate.num_gates}')
