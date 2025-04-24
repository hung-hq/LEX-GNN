import dgl
import pandas as pd
from dgl.data import DGLDataset
import torch as th
import random


class MyFraudDataset(DGLDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False, **kwargs):
        super(MyFraudDataset, self).__init__(
            name='online_payments_dataset',
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            **kwargs
        )

    def process(self):
        # 1. Load your raw data
        df = pd.read_csv("D:/Work/SFIN/PS_rel.csv")

        # 2. Create the graph structure (nodes and edges) and re-index nodes
        src_list = list()
        dst_list = list()
        node_label_list = list()
        feature_sequence = list()
        node_ids = {}  # Mapping from original node ID to new sequential ID
        next_node_id = 0
        cnt = 0
        for idx, row in df.iterrows():
            cnt += 1
            if cnt <= 10000:
                start_node_orig = row[':START_ID']
                end_node_orig = row[':END_ID']
                src_label = row['num_flagged_isFraud']
                dst_label = row['num_flagged_isFraud']

                # Re-index source node
                if start_node_orig not in node_ids:
                    node_ids[start_node_orig] = next_node_id
                    next_node_id += 1
                    node_label_list.append(src_label)
                    feature_sequence.append(random.uniform(1, 100))
                src_list.append(node_ids[start_node_orig])


                # Re-index destination node
                if end_node_orig not in node_ids:
                    node_ids[end_node_orig] = next_node_id
                    next_node_id += 1
                    node_label_list.append(dst_label)
                    feature_sequence.append(random.uniform(1, 100))
                dst_list.append(node_ids[end_node_orig])
            else:
                break
        # print(len(src_list))
        # print(len(dst_list))
        src_nodes = th.tensor(src_list)
        dst_nodes = th.tensor(dst_list)
        graph = dgl.graph((src_nodes, dst_nodes))
        node_features = th.tensor(node_label_list)
        graph.ndata['label'] = node_features
        node_features = th.tensor(feature_sequence)
        graph.ndata['feature'] = node_features
    #     # 3. Extract and assign node features
    #     node_features = th.tensor(...)
    #     self.graph.ndata['feat'] = node_features
    #
    #     # 4. (Optional) Extract and assign edge features


        # print(self.graph.ndata['feature'])
        self.graph = graph
        return self.graph
    #     # 5. Extract and assign node labels
    #     node_labels = th.tensor(...)
    #     self.graph.ndata['label'] = node_labels
    #
    #     # 6. Create train/val/test splits (e.g., node masks)
    #     train_mask = th.tensor(...)
    #     val_mask = th.tensor(...)
    #     test_mask = th.tensor(...)
    #     self.graph.ndata['train_mask'] = train_mask
    #     self.graph.ndata['val_mask'] = val_mask
    #     self.graph.ndata['test_mask'] = test_mask
    #
    #     # Store the number of classes if it's a classification task
    #     if hasattr(node_labels, 'max'):
    #         self._num_classes = int(node_labels.max()) + 1
    #     else:
    #         self._num_classes = None
    #
    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph
    #
    # def __len__(self):
    #     return 1
    #
    # def num_classes(self):
    #     return self._num_classes

# Example usage:
if __name__ == '__main__':
    dataset = MyFraudDataset()
    graph = dataset[0]
    # print(dataset)
    # print(graph.ndata['feature'].shape)
    # print(graph.ndata['label'].shape)
    # print(graph.ndata['train_mask'].sum())