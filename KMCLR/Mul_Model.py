import numpy as np
from numpy import random
import pickle
import gc
import datetime
import torch as t
import torch.nn as nn
import torch.utils.data as dataloader
import utils
import Mul_Data
import GNN
from Mul_Par import args
from Utils.TimeLogger import log
from tqdm import tqdm
import Procedure


t.backends.cudnn.benchmark = True
if t.cuda.is_available():
    use_cuda = True
else:
    use_cuda = False
t.autograd.set_detect_anomaly(True)


class Model():
    def __init__(self):

        self.trn_file = args.path + args.dataset + '/trn_'
        self.tst_file = args.path + args.dataset + '/tst_int'

        self.user_num = -1
        self.item_num = -1
        self.behavior_mats = {}
        self.behaviors = []
        self.behaviors_data = {}

        self.train_loss = []
        self.his_hr = []
        self.his_ndcg = []
        gc.collect()

        self.relu = t.nn.ReLU()
        self.sigmoid = t.nn.Sigmoid()
        self.curEpoch = 0

        if args.dataset == 'Tmall':
            self.behaviors_SSL = ['pv', 'fav', 'cart', 'buy']
            self.behaviors = ['pv', 'fav', 'cart', 'buy']

        elif args.dataset == 'Yelp':
            self.behaviors = ['tip', 'neg', 'neutral', 'pos']
            self.behaviors_SSL = ['tip', 'neg', 'neutral', 'pos']

        elif args.dataset == 'retail':
            self.behaviors = ['fav', 'cart', 'buy']
            self.behaviors_SSL = ['fav', 'cart', 'buy']

        for i in range(0, len(self.behaviors)):
            with open(self.trn_file + self.behaviors[i], 'rb') as fs:
                data = pickle.load(fs)
                self.behaviors_data[i] = data

                if data.get_shape()[0] > self.user_num:
                    self.user_num = data.get_shape()[0]
                if data.get_shape()[1] > self.item_num:
                    self.item_num = data.get_shape()[1]

                if self.behaviors[i] == args.target:
                    self.trainMat = data
                    self.trainLabel = 1 * (self.trainMat != 0)
                    self.labelP = np.squeeze(np.array(np.sum(self.trainLabel, axis=0)))

        for i in range(0, len(self.behaviors)):
            self.behaviors_data[i] = self.behaviors_data[i][:,: max(self.trainMat.indices)+1]
        self.trainMat = self.trainMat[:,: max(self.trainMat.indices)+1]
        self.item_num = max(self.trainMat.indices) + 1

        for i in range(0, len(self.behaviors)):
            self.behavior_mats[i] = utils.get_use(self.behaviors_data[i])

        train_u, train_v = self.trainMat.nonzero()
        train_data = np.hstack((train_u.reshape(-1, 1), train_v.reshape(-1, 1))).tolist()
        train_dataset = Mul_Data.RecDataset_beh(self.behaviors, train_data, self.item_num, self.behaviors_data, True)
        self.train_loader = dataloader.DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0,
                                                  pin_memory=True)

        with open(self.tst_file, 'rb') as fs:
            data = pickle.load(fs)

        test_user = np.array([idx for idx, i in enumerate(data) if i is not None])
        test_item = np.array([i for idx, i in enumerate(data) if i is not None])
        test_data = np.hstack((test_user.reshape(-1, 1), test_item.reshape(-1, 1))).tolist()
        test_dataset = Mul_Data.RecDataset(test_data, self.item_num, self.trainMat, 0, False)
        self.test_loader = dataloader.DataLoader(test_dataset, batch_size=args.batch, shuffle=False, num_workers=0,
                                                 pin_memory=True)

    def prepareModel(self):
        self.gnn_layer = eval(args.gnn_layer)
        self.hidden_dim = args.hidden_dim
        self.model = GNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
        self.opt = t.optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay)
        self.scheduler = t.optim.lr_scheduler.CyclicLR(self.opt, args.opt_base_lr, args.opt_max_lr, step_size_up=5,
                                                       step_size_down=10, mode='triangular', gamma=0.99, scale_fn=None,
                                                       scale_mode='cycle', cycle_momentum=False, base_momentum=0.8,
                                                       max_momentum=0.9, last_epoch=-1)
        if use_cuda:
            self.model = self.model.cuda()

    def innerProduct(self, u, i, j):
        pred_i = t.sum(t.mul(u, i), dim=1) * args.inner_product_mult
        pred_j = t.sum(t.mul(u, j), dim=1) * args.inner_product_mult
        return pred_i, pred_j

    def SSL(self, user_embeddings, user_step_index):

        def multi_neg_sample_pair_index(batch_index, step_index, embedding1,
                                        embedding2):

            index_set = set(np.array(step_index.cpu()))
            batch_index_set = set(np.array(batch_index.cpu()))
            neg2_index_set = index_set - batch_index_set
            neg2_index = t.as_tensor(np.array(list(neg2_index_set))).long().cuda()
            neg2_index = t.unsqueeze(neg2_index, 0)
            neg2_index = neg2_index.repeat(len(batch_index), 1)
            neg2_index = t.reshape(neg2_index, (1, -1))
            neg2_index = t.squeeze(neg2_index)

            neg1_index = batch_index.long().cuda()
            neg1_index = t.unsqueeze(neg1_index, 1)
            neg1_index = neg1_index.repeat(1, len(neg2_index_set))
            neg1_index = t.reshape(neg1_index, (1, -1))
            neg1_index = t.squeeze(neg1_index)

            neg_score_pre = t.sum(
                compute(embedding1, embedding2, neg1_index, neg2_index).squeeze().view(len(batch_index), -1),
                -1)
            return neg_score_pre

        def compute(x1, x2, neg1_index=None, neg2_index=None, τ=0.05):

            if neg1_index != None:
                x1 = x1[neg1_index]
                x2 = x2[neg2_index]

            N = x1.shape[0]
            D = x1.shape[1]

            x1 = x1
            x2 = x2

            scores = t.exp(t.div(t.bmm(x1.view(N, 1, D), x2.view(N, D, 1)).view(N, 1), np.power(D, 1) + 1e-8))

            return scores

        def single_infoNCE_loss_one_by_one(embedding1, embedding2, step_index):
            N = step_index.shape[0]
            D = embedding1.shape[1]

            pos_score = compute(embedding1[step_index], embedding2[step_index]).squeeze()
            neg_score = t.zeros((N,), dtype=t.float64).cuda()


            steps = int(np.ceil(N / args.SSL_batch))
            for i in range(steps):
                st = i * args.SSL_batch
                ed = min((i + 1) * args.SSL_batch, N)
                batch_index = step_index[st: ed]

                neg_score_pre = multi_neg_sample_pair_index(batch_index, step_index, embedding1, embedding2)
                if i == 0:
                    neg_score = neg_score_pre
                else:
                    neg_score = t.cat((neg_score, neg_score_pre), 0)


            con_loss = -t.log(1e-8 + t.div(pos_score, neg_score + 1e-8))

            assert not t.any(t.isnan(con_loss))
            assert not t.any(t.isinf(con_loss))

            return t.where(t.isnan(con_loss), t.full_like(con_loss, 0 + 1e-8), con_loss)

        user_con_loss_list = []

        SSL_len = int(user_step_index.shape[0] / 10)
        user_step_index = t.as_tensor(
            np.random.choice(user_step_index.cpu(), size=SSL_len, replace=False, p=None)).cuda()

        for i in range(len(self.behaviors_SSL)):
            user_con_loss_list.append(
                single_infoNCE_loss_one_by_one(user_embeddings[-1], user_embeddings[i], user_step_index))

        return user_con_loss_list, user_step_index

    def run(self, Kg_model, contrast_model, optimizer, bpr):

        self.Kg_model = Kg_model
        self.contrast_model = contrast_model
        self.optimizer = optimizer
        self.bpr = bpr
        self.prepareModel()

        cvWait = 0
        self.best_HR = 0
        self.best_NDCG = 0

        self.user_embed = None
        self.item_embed = None
        self.user_embeds = None
        self.item_embeds = None

        for e in range(self.curEpoch, args.epoch + 1):
            self.curEpoch = e

            log("*****************Start epoch: %d ************************" % e)

            Procedure.kg_init_transR(self.Kg_model.kg_dataset, Kg_model, optimizer, index=0)
            Procedure.kg_init_TATEC(self.Kg_model.kg_dataset, Kg_model, optimizer, index=1)

            if args.isJustTest == False:
                epoch_loss = self.trainEpoch()
                self.train_loss.append(epoch_loss)
                print(f"epoch {e / args.epoch},  epoch loss{epoch_loss}")
                self.train_loss.append(epoch_loss)
            else:
                break

            HR, NDCG = self.testEpoch(self.test_loader)
            self.his_hr.append(HR)
            self.his_ndcg.append(NDCG)

            self.scheduler.step()

            if HR > self.best_HR:
                self.best_HR = HR
                self.best_epoch = self.curEpoch
                cvWait = 0
                print("----------------------------------------------------------------------------------------------------best_HR",self.best_HR)

            if NDCG > self.best_NDCG:
                self.best_NDCG = NDCG
                self.best_epoch = self.curEpoch
                cvWait = 0
                print("----------------------------------------------------------------------------------------------------best_NDCG",self.best_NDCG)

            if (HR < self.best_HR) and (NDCG < self.best_NDCG):
                cvWait += 1

            if cvWait == args.patience:
                print(f"Early stop at {self.best_epoch} :  best HR: {self.best_HR}, best_NDCG: {self.best_NDCG} \n")
                break

        HR, NDCG = self.testEpoch(self.test_loader)
        self.his_hr.append(HR)
        self.his_ndcg.append(NDCG)

    def negSamp(self, temLabel, sampSize, nodeNum):
        negset = [None] * sampSize
        cur = 0
        while cur < sampSize:
            rdmItm = np.random.choice(nodeNum)
            if temLabel[rdmItm] == 0:
                negset[cur] = rdmItm
                cur += 1
        return negset

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds.cpu()].toarray()
        batch = len(batIds)
        user_id = []
        item_id_pos = []
        item_id_neg = []

        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(labelMat.shape[1])]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = self.negSamp(temLabel[i], sampNum, labelMat.shape[1])

            for j in range(sampNum):
                user_id.append(batIds[i].item())
                item_id_pos.append(poslocs[j].item())
                item_id_neg.append(neglocs[j])
                cur += 1

        return t.as_tensor(np.array(user_id)).cuda(), t.as_tensor(np.array(item_id_pos)).cuda(), t.as_tensor(
            np.array(item_id_neg)).cuda()

    def trainEpoch(self):

        contrast_views = self.contrast_model.get_ui_kg_view()
        log("Drop done")
        Procedure.BPR_train_contrast(self.Kg_model.dataset, self.Kg_model, self.bpr, self.contrast_model, contrast_views, self.optimizer, neg_k=1)

        train_loader = self.train_loader
        time = datetime.datetime.now()
        print("start_ng_samp:  ", time)
        train_loader.dataset.ng_sample()
        time = datetime.datetime.now()
        print("end_ng_samp:  ", time)

        epoch_loss = 0

        self.behavior_loss_list = [None] * len(self.behaviors)

        self.user_id_list = [None] * len(self.behaviors)
        self.item_id_pos_list = [None] * len(self.behaviors)
        self.item_id_neg_list = [None] * len(self.behaviors)


        cnt = 0
        for user, item_i, item_j in tqdm(train_loader):

            user = user.long().cuda()
            self.user_step_index = user

            mul_behavior_loss_list = [None] * len(self.behaviors)
            mul_user_index_list = [None] * len(self.behaviors)

            mul_model = GNN.myModel(self.user_num, self.item_num, self.behaviors, self.behavior_mats).cuda()
            mul_opt = t.optim.AdamW(mul_model.parameters(), lr=args.lr, weight_decay=args.opt_weight_decay)
            mul_model.load_state_dict(self.model.state_dict())

            mul_user_embed, mul_item_embed, mul_user_embeds, mul_item_embeds = mul_model()

            for index in range(len(self.behaviors)):
                not_zero_index = np.where(item_i[index].cpu().numpy() != -1)[0]

                self.user_id_list[index] = user[not_zero_index].long().cuda()

                mul_user_index_list[index] = self.user_id_list[index]
                self.item_id_pos_list[index] = item_i[index][not_zero_index].long().cuda()
                self.item_id_neg_list[index] = item_j[index][not_zero_index].long().cuda()

                mul_userEmbed = mul_user_embed[self.user_id_list[index]]
                mul_posEmbed = mul_item_embed[self.item_id_pos_list[index]]
                mul_negEmbed = mul_item_embed[self.item_id_neg_list[index]]

                mul_pred_i, mul_pred_j = self.innerProduct(mul_userEmbed, mul_posEmbed, mul_negEmbed)

                mul_behavior_loss_list[index] = - (mul_pred_i.view(-1) - mul_pred_j.view(-1)).sigmoid().log()

            mul_infoNCELoss_list, SSL_user_step_index = self.SSL(mul_user_embeds, self.user_step_index)

            for i in range(len(self.behaviors)):
                mul_infoNCELoss_list[i] = (mul_infoNCELoss_list[i]).sum()
                mul_behavior_loss_list[i] = (mul_behavior_loss_list[i]).sum()

            mul_bprloss = sum(mul_behavior_loss_list) / len(mul_behavior_loss_list)
            mul_infoNCELoss = sum(mul_infoNCELoss_list) / len(mul_infoNCELoss_list)
            mul_regLoss = (t.norm(mul_userEmbed) ** 2 + t.norm(mul_posEmbed) ** 2 + t.norm(mul_negEmbed) ** 2)

            mul_model_loss = (mul_bprloss + args.reg * mul_regLoss + args.beta * mul_infoNCELoss) / args.batch

            epoch_loss = epoch_loss + mul_model_loss.item()

            mul_opt.zero_grad(set_to_none=True)
            mul_model_loss.backward()
            nn.utils.clip_grad_norm_(mul_model.parameters(), max_norm=20, norm_type=2)
            mul_opt.step()

            user_embed, item_embed, user_embeds, item_embeds = self.model()

            with t.no_grad():
                user_embed1, item_embed1 = self.Kg_model.getAll()

            user_embed = 0.9*user_embed + 0.1*user_embed1
            item_embed = 0.9*item_embed + 0.1*item_embed1

            for index in range(len(self.behaviors)):
                userEmbed = user_embed[self.user_id_list[index]]
                posEmbed = item_embed[self.item_id_pos_list[index]]
                negEmbed = item_embed[self.item_id_neg_list[index]]

                pred_i, pred_j = self.innerProduct(userEmbed, posEmbed, negEmbed)

                self.behavior_loss_list[index] = - (pred_i.view(-1) - pred_j.view(-1)).sigmoid().log()

            infoNCELoss_list, SSL_user_step_index = self.SSL(user_embeds, self.user_step_index)

            for i in range(len(self.behaviors)):
                infoNCELoss_list[i] = (infoNCELoss_list[i]).sum()
                self.behavior_loss_list[i] = (self.behavior_loss_list[i]).sum()


            bprloss = sum(self.behavior_loss_list) / len(self.behavior_loss_list)
            infoNCELoss = sum(infoNCELoss_list) / len(infoNCELoss_list)
            regLoss = (t.norm(userEmbed) ** 2 + t.norm(posEmbed) ** 2 + t.norm(negEmbed) ** 2)

            loss = (bprloss + args.reg * regLoss + args.beta * infoNCELoss) / args.batch

            epoch_loss = epoch_loss + loss.item()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=20, norm_type=2)
            self.opt.step()

            cnt += 1

        return epoch_loss

    def testEpoch(self, data_loader):

        epochHR, epochNDCG = [0] * 2
        with t.no_grad():
            user_embed, item_embed, user_embeds, item_embeds = self.model()
            user_embed1, item_embed1 = self.Kg_model.getAll()

        user_embed = 0.9 * user_embed + 0.1 * user_embed1
        item_embed = 0.9 * item_embed + 0.1 * item_embed1

        cnt = 0
        tot = 0
        for user, item_i in data_loader:
            user_compute, item_compute, user_item1, user_item100 = self.sampleTestBatch(user, item_i)
            userEmbed = user_embed[user_compute]
            itemEmbed = item_embed[item_compute]

            pred_i = t.sum(t.mul(userEmbed, itemEmbed), dim=1)

            hit, ndcg = self.calcRes(t.reshape(pred_i, [user.shape[0], 100]), user_item1, user_item100)
            epochHR = epochHR + hit
            epochNDCG = epochNDCG + ndcg
            cnt += 1
            tot += user.shape[0]

        result_HR = epochHR / tot
        result_NDCG = epochNDCG / tot
        print(f"Step {cnt}:  hit:{result_HR}, ndcg:{result_NDCG}")

        return result_HR, result_NDCG

    def calcRes(self, pred_i, user_item1, user_item100):

        hit = 0
        ndcg = 0

        for j in range(pred_i.shape[0]):

            _, shoot_index = t.topk(pred_i[j], args.shoot)
            shoot_index = shoot_index.cpu()
            shoot = user_item100[j][shoot_index]
            shoot = shoot.tolist()

            if type(shoot) != int and (user_item1[j] in shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(shoot.index(user_item1[j]) + 2))
            elif type(shoot) == int and (user_item1[j] == shoot):
                hit += 1
                ndcg += np.reciprocal(np.log2(0 + 2))

        return hit, ndcg

    def sampleTestBatch(self, batch_user_id, batch_item_id):

        batch = len(batch_user_id)
        tmplen = (batch * 100)

        sub_trainMat = self.trainMat[batch_user_id].toarray()
        user_item1 = batch_item_id
        user_compute = [None] * tmplen
        item_compute = [None] * tmplen
        user_item100 = [None] * (batch)

        cur = 0
        for i in range(batch):
            pos_item = user_item1[i]
            negset = np.reshape(np.argwhere(sub_trainMat[i] == 0), [-1])

            random_neg_sam = np.random.permutation(negset)[:99]
            user_item100_one_user = np.concatenate((random_neg_sam, np.array([pos_item])))
            user_item100[i] = user_item100_one_user

            for j in range(100):
                user_compute[cur] = batch_user_id[i]
                item_compute[cur] = user_item100_one_user[j]
                cur += 1

        return user_compute, item_compute, user_item1, user_item100
