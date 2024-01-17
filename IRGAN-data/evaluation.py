from scipy.sparse import csr_matrix
from scipy.stats import entropy
import numpy as np

def data_extract(workdir):
    users_list,items_list=[],[]
    with open( workdir)as fin:
        for line in fin:
            line = line.split()
            uid = int(line[0])
            iid = int(line[1])
            users_list.append(uid)
            items_list.append(iid)
    num_users = max(users_list)+1
    num_items = max(items_list)+1
    return users_list,items_list,num_users,num_items

def build_csr_matrix(users_list,items_list,num_users,num_items):
    implicit_rating =[]
    for i in range(len(users_list)):
        implicit_rating.append(1)
    matrix_implicit = csr_matrix((implicit_rating, (users_list, items_list)), shape=(num_users, num_items))
    return matrix_implicit

def get_item_distribution(profiles):
    # [min(max(0, round(i)), 5) for i in a]
    profiles_T = profiles.transpose()
    fn_count = lambda item_vec: np.array(
    [sum([(j == i) for j in item_vec]) for i in range(2)]) / item_vec.shape[0]
    fn_norm = lambda item_vec: np.asarray(item_vec) / np.sum(item_vec)
    item_distribution = np.array(list(map(fn_count, profiles_T)))
    item_distribution = np.array(list(map(fn_norm, item_distribution)))
    return item_distribution

def eval_TVD_JS(P, Q):
    # TVD
    dis_TVD = np.mean(np.sum(np.abs(P - Q) / 2, 1))
    # JS
    fn_KL = lambda p, q: entropy(p, q)
    M = (P + Q) / 2
    js_vec = []
    for iid in range(P.shape[0]):
        p, q, m = P[iid], Q[iid], M[iid]
        js_vec.append((fn_KL(p, m) + fn_KL(q, m)) / 2)
    dis_JS = np.mean(np.array(js_vec))
    return dis_TVD, dis_JS


users_list, items_list =[],[]
users_list,items_list,num_users,num_items = data_extract( './data/ml-100k/train1.txt')
syn_users,syn_items,syn_users_num,num_items = data_extract( './data/ml-100k-syn10/train1.txt')

matrix_implicit = build_csr_matrix(users_list,items_list,num_users,num_items)
matrix_implicit_syn = build_csr_matrix(syn_users,syn_items,syn_users_num,num_items)
real_item_distribution = get_item_distribution(matrix_implicit.toarray())
fake_gan_distribution = get_item_distribution(matrix_implicit_syn.toarray())
dis_TVD, dis_JS = eval_TVD_JS(real_item_distribution, fake_gan_distribution)
print(dis_TVD)
print(dis_JS)
