import torch
from model import Generator
from config import irgan_config
from data_utils import RecDataset, DataProvider
from evaluation.rec_evaluator import RecEvaluator

epochs = irgan_config.epochs
batch_size = irgan_config.batch_size
epochs_d = irgan_config.epochs_d
epochs_g = irgan_config.epochs_g
emb_dim = irgan_config.emb_dim
eta_G = irgan_config.eta_G
eta_D = irgan_config.eta_D
device = irgan_config.device
weight_decay_g = irgan_config.weight_decay_g
weight_decay_d = irgan_config.weight_decay_d
# Hyper-parameters and datset-specific parameters
rec_dataset = RecDataset(irgan_config.dir_path)
all_users = rec_dataset.get_users()
all_items = rec_dataset.get_items()
num_users = rec_dataset.get_num_users()
num_items = rec_dataset.get_num_items()
bought_mask = rec_dataset.get_bought_mask().to(device)
eval_dict = rec_dataset.get_interaction_records("test")
train_ui = rec_dataset.get_user_item_pairs()
dp = DataProvider(device)
evaluator = RecEvaluator(eval_dict, None, device)

G = Generator(num_users, num_items, emb_dim, bought_mask)
G = G.to(device)

pretrained_model = torch.load("./pretrained_models/pretrained_model_dns.pkl",map_location=irgan_config.device)
G.load_state_dict(pretrained_model)

fake_users,fake_items = dp.geneate_synthesis_data(G, train_ui,  fake_users_num= 50)

print(len(fake_users))
print(fake_users[0])
print(len(fake_items))

