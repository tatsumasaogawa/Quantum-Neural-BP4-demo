# %%
#Import external libraries
import torch
import sys
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import random
import matplotlib.pylab as plt

# %%
class NBP_2_oc(nn.Module):
    def __init__(self, n: int, k: int, m: int, m1: int, m2: int, codeType: str, n_iterations: int,
                 folder_weights: str = None,
                 batch_size: int = 1):
        super().__init__()
        self.name = "Neural_BP_2_Decoder"
        self.batch_size = batch_size
        self.codeType = codeType
        self.n = n # physical qubit の数
        self.k = k # logical qubit の数
        #m_oc is the number rows of the overcomplete check matrix
        self.m_oc = m
        self.m1 = m1
        self.m2 = m2
        #m is the number of rows of the full rank check matrix
        self.m = n - k
        #If True, then all outgoing edges on the same CN has the same weight, configurable
        self.one_weight_per_cn = True
        self.rate = self.k / self.n
        self.n_iterations = n_iterations # BP の iter数
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.xhat = torch.zeros((batch_size, self.n))
        self.zhat = torch.zeros((batch_size, self.n))
        self.load_matrices()

        if not folder_weights:
            #initilize weights with 1 if none given
            self.ini_weight_as_one(n_iterations)
        else:
            # load pretrained weights stored in directory "folder":
            self.load_weights(folder_weights, self.device)

    def fx(self, a: torch.tensor, b: torch.tensor) -> torch.Tensor:
        # ln(exp(x)+exp(y)) = max(x,y)+ln(1+exp(-|x-y|)
        return torch.max(a, b) + self.log1pexp(-1 * torch.abs(a - b))

    def log1pexp(self, x):
        # more stable version of log(1 + exp(x))
        m = nn.Softplus(beta=1, threshold=50)
        return m(x)

    def calculate_self_syn(self):
        # z ゲートが x エラーに反応するので, Hz * errorx
        
        # errorx, errorz: (batch_size, n)
        # Hx, Hz: (1, m1 or m2, n)
        # synx, synz: (batch_size, m1 or m2, 1)
        
        self.synx = torch.matmul(self.Hz, torch.transpose(self.errorx, 0, 1)) # transpose: 軸を入れ替える
        self.synz = torch.matmul(self.Hx, torch.transpose(self.errorz, 0, 1))
        self.synx = torch.remainder(torch.transpose(self.synx, 2, 0), 2) # remainder: 余り
        self.synz = torch.remainder(torch.transpose(self.synz, 2, 0), 2)
        return torch.cat((self.synz, self.synx), dim=1)


    # BP_2 用に改修済
    def loss(self, Gamma) -> torch.Tensor:
        """loss functions proposed in [1] eq. 11"""
        # Gamma: (batch_size, 2n)

        # first row, anti-commute with X, second row, anti-commute with Z, [1] eq. 10
        # 1列目、Xとのアンチコミュート、2列目、Zとのアンチコミュート [1] 式10
        prob = torch.sigmoid(-1.0 * Gamma).float() # (batch_size, 2n)

        prob_aX = prob[:, :self.n] # (batch_size, n)
        prob_aZ = prob[:, self.n:]

        # assert not torch.isinf(prob_aX).any()
        # assert not torch.isinf(prob_aZ).any()
        # assert not torch.isnan(prob_aX).any()
        # assert not torch.isnan(prob_aZ).any()

        #Depend on if the error commute with the entries in S_dual, which is denoted as G here
        #CSS constructions gives the simplification that Gx contains only X entries, and Gz contains on Z
        #誤差がS_dualのエントリと一致するかどうかに依存する。
        #GxはXの項目のみを含み、GzはZの項目のみを含む。 
        # error: (batch_size, n)
        correctionx = torch.zeros_like(self.errorx)
        correctionz = torch.zeros_like(self.errorz)

        correctionz[self.qx == 1] = prob_aX[self.qx == 1]
        # correctionz[self.qz == 1] = 1 - prob_aX[self.qz == 1]
        # correctionz[self.qy == 1] = 1 - prob_aX[self.qy == 1]
        # correctionz[self.qi == 1] = prob_aX[self.qi == 1]

        correctionx[self.qz == 1] = prob_aZ[self.qz == 1]
        # correctionx[self.qx == 1] = 1 - prob_aZ[self.qx == 1]
        # correctionx[self.qy == 1] = 1 - prob_aZ[self.qy == 1]
        # correctionx[self.qi == 1] = prob_aZ[self.qi == 1]

        #first summ up the probability of anti-commute for all elements in each row of G
        #まず、Gの各行の全要素について、アンチコミュートの確率を合計する。
        # たぶん G = S ^ ⊥: (1, m, n)
        synx = torch.matmul(self.Gz, torch.transpose(correctionx.float(), 0, 1)) # (1, m, batch_size)
        synz = torch.matmul(self.Gx, torch.transpose(correctionz.float(), 0, 1))
        synx = torch.transpose(synx, 2, 0) # (batch_size, m, 1)
        synz = torch.transpose(synz, 2, 0)
        syn_real = torch.cat((synz, synx), dim=1) # (batch_size, 2m, 1)

        #the take the sin function, then summed up for all rows of G
        #sin関数をとり、Gのすべての行について合計する。
        loss = torch.zeros(1, self.batch_size)
        for b in range(self.batch_size):
            loss[0, b] = torch.sum(torch.abs(torch.sin(np.pi / 2 * syn_real[b, :, :])))

        assert not torch.isnan(loss).any()
        assert not torch.isinf(loss).any()

        return loss # (1, batch_size)
    

    # BP_2 用に改修済
    def variable_node_update(self, incoming_messages, llr, weights_vn, weights_llr):
        # As we deal with CSS codes, all non-zero entries on the upper part anti-commute with Z and Y and commute with X
        # all non-zero entries on the upper part anti-commute with X and Y and commute with Z
        # Then the calculation can be done in matrices => speed up training (probably)
        # CSSコードを扱うので、上部の非ゼロエントリーはすべてZとYと反共約し、Xと共約する。
        # 上部のすべての非ゼロエントリは X および Y と反共約し、Z と共約する。
        # すると計算は行列でできる => トレーニングのスピードアップ（たぶん）

        # incoming_messages (batch_size, m, n)
        # (batch_size, m, 2n) であるべき
        # incoming_messages_upper = incoming_messages[:, 0:self.m1, :]
        # incoming_messages_lower = incoming_messages[:, self.m1:self.m_oc, :]
        # incoming_messages_upper = incoming_messages_upper.to(self.device)
        # incoming_messages_lower = incoming_messages_lower.to(self.device)
        
        # (batch_size, 1, n)
        # Gammaz = llr * weights_llr + torch.sum(incoming_messages_upper, dim=1, keepdim=True)
        # Gammax = llr * weights_llr + torch.sum(incoming_messages_lower, dim=1, keepdim=True)
        
        # print('variable node update')
        # incoming_messages (batch_size, m, 2n)
        # Gamma: (batch_size, 1, 2n)
        Gamma = llr * weights_llr + torch.sum(incoming_messages, dim=1, keepdim=True)
        # print('gamma ok')

        # double に変換
        # Gammaz = Gammaz.double().to(self.device)
        # Gammax = Gammax.double().to(self.device)

        # double に変換
        Gamma = Gamma.double().to(self.device) # (batch_size, 1, 2n)
        # print('double ok')
        
        #can be re-used for hard-decision in decoding, but not used in training as we don't check for decoding success
        #we are only interested in the loss during training
        #デコード時のハード判定に再利用できるが、デコードの成功をチェックしないため、トレーニングでは使用しない。
        #我々はトレーニング中の損失にしか興味がない。
        # (batch_size, 1, 2n)
        # Gamma = torch.cat((Gammax, Gammaz), dim=2).to(self.device)

        # assert not torch.isinf(Gammaz).any()
        # assert not torch.isinf(Gammax).any()
        # assert not torch.isinf(Gammay).any()

        # f(x, y) = ln(exp(x)+exp(y)) = max(x,y)+ln(1+exp(-|x-y|)
        # outgoing_messages_upper = self.log1pexp(-1.0 * Gammax) - self.fx(-1.0 * Gammaz, -1.0 * Gammay)
        # outgoing_messages_lower = self.log1pexp(-1.0 * Gammaz) - self.fx(-1.0 * Gammax, -1.0 * Gammay)
        # Gamma_all = torch.cat((outgoing_messages_upper, outgoing_messages_lower), dim=1).to(self.device)

        # sigma の下の <zeta, S_ji> = 1 に対応
        # 0 のところが消える
        # outgoing_messages_upper = outgoing_messages_upper * self.Hx
        # outgoing_messages_lower = outgoing_messages_lower * self.Hz
        # outgoing_messages = torch.cat((outgoing_messages_upper, outgoing_messages_lower), dim=1)

        # N(v) \ c に対応して, incoming_messages を引く
        outgoing_messages = Gamma - incoming_messages # (batch_size, m, 2n)
        # print('sub incoming messages ok')
        # sigma の下の <zeta, S_ji> = 1 に対応
        # outgoing_messages = outgoing_messages * self.H

        # assert not torch.isinf(Gammaz).any()
        # assert not torch.isinf(Gammax).any()
        # assert not torch.isinf(Gammay).any()

        #to avoid numerical issues
        #数値的な問題を避けるため
        outgoing_messages = torch.clip(outgoing_messages, -30.0, 30.0)
        # print('clip ok')

        # あらかじめ, weights_vn を掛けた状態で返す
        # weights_vn: (1, m, n)
        return outgoing_messages.float() * weights_vn, torch.squeeze(Gamma) #  out_going_messages: (batch_size, m, 2n), Gamma: (batch_size, 2n)

    # コードの意味は分からないが, BP_2 と BP_4 で特に改修は必要ない
    def check_node_update(self, incoming_messages: torch.Tensor, weights_cn: torch.Tensor) -> torch.Tensor:
        multipicator = torch.pow(-1, self.syn)
        multipicator = multipicator * self.H

        # use the simplification with the phi function to turn multipilication to addtion
        # a bit more troublesome than the usual SPA, because want to do it in matrix
        # 乗算を加算に変えるために phi 関数を使った単純化を使う
        # 通常の SPA よりも少し面倒である。
        # SPA: Sum-Product Algorithm

        # incoming_messages: (batch_size, m, 2n)
        incoming_messages_sign = torch.sign(incoming_messages) # torch.sign: 1. or -1. or 0. を返す
        incoming_messages_sign[incoming_messages == 0] = 1
        first_part = torch.prod(incoming_messages_sign, dim=2, keepdim=True) # 2 番目の次元の積をとる (batch_size, m, 1)
        # print('prod ok')
        # print(first_part.size())
        first_part = first_part * self.H # (batch_size, m, 2n)
        first_part = first_part / incoming_messages_sign # i' in N(j) \ {i} に対応して, incoming_messages_sign で割る
        first_part = self.H * first_part # (batch_size, m, 2n)
        assert not torch.isinf(first_part).any()
        assert not torch.isnan(first_part).any()

        incoming_messages_abs = torch.abs(incoming_messages).double()
        helper = torch.ones_like(incoming_messages_abs)
        helper[incoming_messages_abs == 0] = 0
        incoming_messages_abs[incoming_messages == 0] = 1.0

        phi_incoming_messages = -1.0 * torch.log(torch.tanh(incoming_messages_abs / 2.0))
        phi_incoming_messages = phi_incoming_messages * helper
        phi_incoming_messages = phi_incoming_messages * self.H

        temp = torch.sum(phi_incoming_messages, dim=2, keepdim=True)
        Aij = temp * self.H

        sum_msg = Aij - phi_incoming_messages
        helper = torch.ones_like(sum_msg)
        helper[sum_msg == 0] = 0
        sum_msg[sum_msg == 0] = 1.0

        second_part = -1 * torch.log(torch.tanh(sum_msg / 2.0))
        second_part = second_part * helper
        second_part = second_part * self.H
        assert not torch.isinf(second_part).any()
        assert not torch.isnan(second_part).any()

        outgoing_messages = first_part * second_part
        outgoing_messages = outgoing_messages * multipicator

        outgoing_messages = (outgoing_messages * weights_cn).float()
        return outgoing_messages

    # BP_2 用に改修済
    def forward(self, errorx: torch.Tensor, errorz: torch.Tensor, ep: float, batch_size=1) -> torch.Tensor:
        """main decoding procedure"""
        loss_array = torch.zeros(self.batch_size, self.n_iterations).float().to(self.device)
        
        # batch_size != self.batch_size のときに例外を投げる
        assert batch_size == self.batch_size

        self.errorx = errorx.to(self.device)
        self.errorz = errorz.to(self.device)

        # 2 次信念伝播なので 2 変数
        self.qx = torch.zeros_like(self.errorx)
        self.qz = torch.zeros_like(self.errorx)
        # self.qy = torch.zeros_like(self.errorx)
        # self.qi = torch.ones_like(self.errorx)

        # x エラーの箇所を 1 にする
        self.qx[self.errorx == 1] = 1
        # self.qx[self.errorz == 1] = 0

        # z エラーの箇所を 1 にする
        self.qz[self.errorz == 1] = 1
        # self.qz[self.errorx == 1] = 0

        # 一旦 z エラーの箇所を y エラーとしてから, x エラーが無い箇所を除く
        # self.qy[self.errorz == 1] = 1
        # self.qy[self.errorx != self.errorz] = 0

        # qi は初め 1 で初期化していることに注意
        # エラーのある箇所を除く
        # self.qi[self.errorx == 1] = 0
        # self.qi[self.errorz == 1] = 0

        self.syn = self.calculate_self_syn()

        #initial LLR to, first equation in [1,Sec.II-C]
        # physical error rate を ep とした llr
        llr = np.log((1 - ep) / ep)

        messages_cn_to_vn = torch.zeros((batch_size, self.m_oc, 2 * self.n)).to(self.device)
        self.batch_size = batch_size

        # initlize VN message
        messages_vn_to_cn, _ = self.variable_node_update(messages_cn_to_vn, llr, self.weights_vn[0],
                                                            self.weights_llr[0])
        # print('variable node update ok')

        # iteratively decode, decode will continue till the max. iteration, even if the syndrome already matched
        # 反復デコードでは、シンドロームがすでにマッチしていても、デコードは最大反復まで続けられる。
        for i in range(self.n_iterations):

            assert not torch.isnan(self.weights_llr[i]).any()
            assert not torch.isnan(self.weights_cn[i]).any()
            assert not torch.isnan(messages_cn_to_vn).any()

            # check node update:
            messages_cn_to_vn = self.check_node_update(messages_vn_to_cn, self.weights_cn[i])
            # print('check node update ok')

            assert not torch.isnan(messages_cn_to_vn).any()
            assert not torch.isinf(messages_cn_to_vn).any()

            # variable node update:
            messages_vn_to_cn, Gamma = self.variable_node_update(messages_cn_to_vn, llr, self.weights_vn[i + 1],
                                                                        self.weights_llr[i + 1])

            assert not torch.isnan(messages_vn_to_cn).any()
            assert not torch.isinf(messages_vn_to_cn).any()
            assert not torch.isnan(Gamma).any()
            assert not torch.isinf(Gamma).any()

            loss_array[:, i] = self.loss(Gamma)


        _, minIdx = torch.min(loss_array, dim=1, keepdim=False)


        loss = torch.zeros(self.batch_size, ).float().to(self.device)
        #take average of the loss for the first iterations till the loss is minimized
        #損失が最小になるまでの、最初の反復の損失の平均をとる。
        for b in range(batch_size):
            for idx in range(minIdx[b] + 1):
                loss[b] += loss_array[b, idx]
            loss[b] /= (minIdx[b] + 1)

        loss = torch.sum(loss, dim=0) / self.batch_size

        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        return loss
    

    # BP_2 用に改修済
    def check_syndrome(self, Tau):
        """performs hard decision to give the estimated error and check for decoding success.
        However, not used in the current script, as we are only performing trainig"""
        """推定誤差を与えるハード判定を行い、デコードの成功をチェックする。
        しかし、今回のスクリプトでは学習のみを行うため、使用しない。"""

        # Tau (Gamma): (batch_size, 2n)

        # tmp はエラーなしに対応する
        # tmp = torch.zeros(self.batch_size, 1, self.n).to(self.device)
        # Tau = torch.cat((tmp, Tau), dim=1)

        # minVal, minIdx = torch.min(Tau, dim=1, keepdim=False)

        # Tau = torch.sigmoid(-1.0 * Tau).float()

        self.xhat = torch.zeros((self.batch_size, self.n)).to(self.device)
        self.zhat = torch.zeros((self.batch_size, self.n)).to(self.device)

        self.xhat[Tau[:, :self.n] < 0] = 1 # (batch_size, n)
        # self.xhat[minIdx == 2] = 1

        self.zhat[Tau[:, self.n:] < 0] = 1
        # self.zhat[minIdx == 3] = 1
        m = torch.nn.ReLU()

        synx = torch.matmul(self.Hz, torch.transpose(self.xhat, 0, 1)) # (1, m, batch_size)
        synz = torch.matmul(self.Hx, torch.transpose(self.zhat, 0, 1))
        synx = torch.transpose(synx, 2, 0) # (batch_size, m, 1)
        synz = torch.transpose(synz, 2, 0)
        synhat = torch.remainder(torch.cat((synz, synx), dim=1), 2) # (batch_size, 2m, 1)

        syn_match = torch.all(torch.all(torch.eq(self.syn, synhat), dim=1), dim=1) # torch.bool

        # correction: エラー訂正後のエラーの有無 (0: エラーなし, 1: エラーあり)
        correctionx = torch.remainder(self.xhat + self.errorx, 2) # (batch_size, n)
        correctionz = torch.remainder(self.zhat + self.errorz, 2)
        synx = torch.matmul(self.Gz, torch.transpose(correctionx, 0, 1)) # (1, m, batch_size)
        synz = torch.matmul(self.Gx, torch.transpose(correctionz, 0, 1))
        synx = torch.transpose(synx, 2, 0) # (batch_size, m, 1)
        synz = torch.transpose(synz, 2, 0)
        self.syn_real = torch.cat((synz, synx), dim=1) # (batch_size, 2m, 1)

        syn_real = torch.remainder(self.syn_real, 2)
        # tmmp = torch.sum(syn_real, dim=1, keepdim=False)
        success = torch.all(torch.eq(torch.sum(syn_real, dim=1, keepdim=False), 0), dim=1) # torch.bool
        return syn_match, success # torch.bool
    
    
    
    def unsqueeze_batches(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Checks if tensor is 2D or 3D. If tensor is 2D, insert extra dimension (first dimension)
        This method can be used to allow decoding of
            batches of codewords (batch size, m, n)
            as well as single codewords (m, n)
        """
        """
        テンソルが2次元か3次元かを調べる。テンソルが2Dの場合、余分な次元（1次元目）を挿入する。
        このメソッドは
            コードワードのバッチ(バッチサイズ、m, n)
            単一のコードワード(m, n)と同様に
        """
        if tensor.dim() == 3:
            return tensor
        elif tensor.dim() == 2:
            return torch.unsqueeze(tensor, dim=0)


    #continue with the NBP_oc class, some tool functions
    #NBP_ocクラスを継続し、いくつかのツール機能を追加する。
    def load_matrices(self):
        """reads in the check matrix for decoding as well as the dual matrix for checking decoding success"""
        """デコードのためのチェックマトリックスと、デコードの成功をチェックするためのデュアルマトリックスを読み込む"""
        file_nameGx = "./PCMs/" + self.codeType + "_" + str(self.n) + "_" + str(
            self.k) + "/" + self.codeType + "_" + str(self.n) + "_" + str(self.k) + "_Gx.alist"
        file_nameGz = "./PCMs/" + self.codeType + "_" + str(self.n) + "_" + str(
            self.k) + "/" + self.codeType + "_" + str(self.n) + "_" + str(self.k) + "_Gz.alist"
        Gx = readAlist(file_nameGx)
        Gz = readAlist(file_nameGz)

        file_nameH = "./PCMs/" + self.codeType + "_" + str(self.n) + "_" + str(
            self.k) + "/" + self.codeType + "_" + str(self.n) + "_" + str(self.k) + "_H_" + str(self.m_oc) + ".alist"

        H = readAlist(file_nameH)
        self.H = H
        Hx = H[0:self.m1, :]
        Hz = H[self.m1:self.m_oc, :]
        Gx = torch.from_numpy(Gx).float()
        Gz = torch.from_numpy(Gz).float()
        Hx = torch.from_numpy(Hx).float()
        Hz = torch.from_numpy(Hz).float()

        O = torch.zeros(self.m1, self.n)


        # first dim for batches.
        self.Hx = self.unsqueeze_batches(Hx).float().to(self.device) # (1, m, n)
        self.Hz = self.unsqueeze_batches(Hz).float().to(self.device)
        self.Gx = self.unsqueeze_batches(Gx).float().to(self.device)
        self.Gz = self.unsqueeze_batches(Gz).float().to(self.device)
        O = self.unsqueeze_batches(O).float().to(self.device)
        # print('except init H ok')

        self.H = torch.cat((torch.cat((self.Hx, O), dim=2), torch.cat((O, self.Hz), dim=2)), dim=1).float().to(self.device) # (1, m, 2n)
        self.H_reverse = 1 - self.H
        # print('init H ok')


    def ini_weight_as_one(self, n_iterations: int):
        """this function can be configured to determine which parameters are trainable"""
        """この関数は、どのパラメーターが訓練可能かを決定するために設定することができる"""
        self.weights_llr = [] # log-likelihood ratio
        self.weights_cn = []
        self.weights_vn = []
        for i in range(n_iterations):
            if self.one_weight_per_cn:
                self.weights_cn.append(torch.ones((1, self.m_oc, 1), requires_grad=True, device=self.device))
            else:
                self.weights_cn.append(torch.ones((1, self.m_oc, 2 * self.n), requires_grad=True, device=self.device))
            self.weights_llr.append(torch.ones((1, 1, 2 * self.n), requires_grad=True, device=self.device))
            self.weights_vn.append(torch.ones(1, self.m_oc, 2 * self.n, requires_grad=False, device=self.device))
        self.weights_vn.append(torch.ones(1, self.m_oc, 2 * self.n, requires_grad=False, device=self.device))
        self.weights_llr.append(torch.ones((1, 1, 2 * self.n), requires_grad=True, device=self.device))
    

    def save_weights(self):
        """weights are saved twice, once as .pt for python, once as .txt for c++"""
        """重みは2回保存されます。1回はpython用の.ptとして、もう1回はc++用の.txtとして保存されます"""
        "./training_results/" + codeType + "_" + str(n) + "_" + str(k) + "_" + str(m) + "/"
        os.makedirs(path, exist_ok=True)
        #some parameters may not be trained, but we save them anyway
        file_vn = "weights_vn.pt"
        file_cn = "weights_cn.pt"
        file_llr = "weights_llr.pt"

        torch.save(self.weights_vn, os.path.join(path, file_vn))
        torch.save(self.weights_cn, os.path.join(path, file_cn))
        torch.save(self.weights_llr, os.path.join(path, file_llr))
        print(f'  weights saved to {file_cn},{file_vn}, and {file_llr}.\n')

        # the following codes save the weights into txt files, which is used for C++ code for evaluating the trained
        # decoder. So the C++ codes don't need to mess around with python packages
        # not very elegant but will do for now
        # 以下のコードは重みをtxtファイルに保存する。
        # C++ コードに使用される。そのため、C++のコードはpythonのパッケージをいじる必要がない。
        # あまりエレガントではないが、今のところはこれで十分だろう
        if sys.version_info[0] == 2:
            import cStringIO
            StringIO = cStringIO.StringIO
        else:
            import io

        StringIO = io.StringIO

        # write llr weights, easy
        f = open(path + "weight_llr.txt", "w")
        with StringIO() as output:
            output.write('{}\n'.format(len(self.weights_llr)))
            for i in self.weights_llr:
                data = i.detach().cpu().numpy().reshape(2 * self.n, 1)
                opt = ["%.16f" % i for i in data]
                output.write(' '.join(opt))
                output.write('\n')
            f.write(output.getvalue())
        f.close()

        # write CN weights
        H_tmp = self.H.detach().cpu().numpy().reshape(self.m_oc, 2 * self.n)
        H_tmp = np.array(H_tmp, dtype='int')
        f = open(path + "weight_cn.txt", "w")
        with StringIO() as output:
            output.write('{}\n'.format(len(self.weights_cn)))
            nRows, nCols = H_tmp.shape
            # first line: matrix dimensions
            output.write('{} {}\n'.format(nCols, nRows))

            # next three lines: (max) column and row degrees
            colWeights = H_tmp.sum(axis=0)
            rowWeights = H_tmp.sum(axis=1)

            maxRowWeight = max(rowWeights)

            if self.one_weight_per_cn:
                # column-wise nonzeros block
                for i in self.weights_cn:
                    matrix = i.detach().cpu().numpy().reshape(self.m_oc, 1)
                    for rowId in range(nRows):
                        opt = ["%.16f" % i for i in matrix[rowId]]
                        for i in range(rowWeights[rowId].astype('int') - 1):
                            output.write(opt[0])
                            output.write(' ')
                        output.write(opt[0])
                        # fill with zeros so that every line has maxDegree number of entries
                        output.write(' 0' * (maxRowWeight - rowWeights[rowId] - 1).astype('int'))
                        output.write('\n')
            else:
                # column-wise nonzeros block
                for i in self.weights_cn:
                    matrix = i.detach().cpu().numpy().reshape(self.m_oc, 2 * self.n)
                    matrix *= self.H[0].detach().cpu().numpy().reshape(self.m_oc, 2 * self.n)
                    for rowId in range(nRows):
                        nonzeroIndices = np.flatnonzero(matrix[rowId, :])  # AList uses 1-based indexing
                        output.write(' '.join(map(str, matrix[rowId, nonzeroIndices])))
                        # fill with zeros so that every line has maxDegree number of entries
                        output.write(' 0' * (maxRowWeight - len(nonzeroIndices)))
                        output.write('\n')
            f.write(output.getvalue())
        f.close()

        # write VN weights
        H_tmp = self.H.detach().cpu().numpy().reshape(self.m_oc, 2 * self.n)
        H_tmp = np.array(H_tmp, dtype='int')
        f = open(path + "weight_vn.txt", "w")
        with StringIO() as output:
            output.write('{}\n'.format(len(self.weights_vn)))
            nRows, nCols = H_tmp.shape
            # first line: matrix dimensions
            output.write('{} {}\n'.format(nCols, nRows))

            # next three lines: (max) column and row degrees
            colWeights = H_tmp.sum(axis=0)
            rowWeights = H_tmp.sum(axis=1)

            maxColWeight = max(colWeights)

            # column-wise nonzeros block
            for i in self.weights_vn:
                matrix = i.detach().cpu().numpy().reshape(self.m_oc, 2 * self.n)
                matrix *= self.H[0].detach().cpu().numpy().reshape(self.m_oc, 2 * self.n)
                for colId in range(nCols):
                    nonzeroIndices = np.flatnonzero(matrix[:, colId])  # AList uses 1-based indexing
                    output.write(' '.join(map(str, matrix[nonzeroIndices, colId])))
                    # fill with zeros so that every line has maxDegree number of entries
                    output.write(' 0' * (maxColWeight - len(nonzeroIndices)))
                    output.write('\n')
            f.write(output.getvalue())
        f.close()

# %%
#helper functions
def readAlist(directory):
    '''
    Reads in a parity check matrix (pcm) in A-list format from text file. returns the pcm in form of a numpy array with 0/1 bits as float64.
    テキストファイルからAリスト形式のパリティチェック行列(pcm)を読み込み、float64として0/1ビットのnumpy配列形式でpcmを返す
    '''

    alist_raw = []
    with open(directory, "r") as f:
        lines = f.readlines()
        for line in lines:
            # remove trailing newline \n and split at spaces:
            line = line.rstrip().split(" ")
            # map string to int:
            line = list(map(int, line))
            alist_raw.append(line)
    alist_numpy = alistToNumpy(alist_raw)
    alist_numpy = alist_numpy.astype(float)
    return alist_numpy


def alistToNumpy(lines):
    '''Converts a parity-check matrix in AList format to a 0/1 numpy array'''
    nCols, nRows = lines[0]
    if len(lines[2]) == nCols and len(lines[3]) == nRows:
        startIndex = 4
    else:
        startIndex = 2
    matrix = np.zeros((nRows, nCols), dtype=float)
    for col, nonzeros in enumerate(lines[startIndex:startIndex + nCols]):
        for rowIndex in nonzeros:
            if rowIndex != 0:
                matrix[rowIndex - 1, col] = 1
    return matrix

# %%
def optimization_step(decoder, ep0, optimizer: torch.optim.Optimizer, errorx, errorz):
    #call the forward function
    # print('optimization_step')
    loss = decoder(errorx, errorz, ep0, batch_size=batch_size)
    # print('forward ok')

    # delete old gradients.
    optimizer.zero_grad()
    # calculate gradient
    loss.backward()
    # update weights
    optimizer.step()

    return loss.detach()


def training_loop(decoder, optimizer, r1, r2, ep0, num_batch, path):
    print(f'training on random errors, weight from {r1} to {r2} ')
    loss_length = num_batch
    loss = torch.zeros(loss_length)


    idx = 0
    with tqdm(total=loss_length) as pbar:
        for i_batch in range(num_batch):
            errorx = torch.tensor([])
            errorz = torch.tensor([])
            for w in range(r1, r2):
                ex, ez = addErrorGivenWeight(decoder.n, w, batch_size // (r2 - r1 + 1))
                errorx = torch.cat((errorx, ex), dim=0)
                errorz = torch.cat((errorz, ez), dim=0)
            res_size = batch_size - ((batch_size // (r2 - r1 + 1)) * (r2 - r1))
            ex, ez = addErrorGivenWeight(decoder.n, r2, res_size)
            errorx = torch.cat((errorx, ex), dim=0)
            errorz = torch.cat((errorz, ez), dim=0)

            loss[idx]= optimization_step(decoder, ep0, optimizer, errorx, errorz)
            pbar.update(1)
            pbar.set_description(f"loss {loss[idx]}")
            idx += 1
        decoder.save_weights()

    print('Training completed.\n')
    return loss

def plot_loss(loss, path, myrange = 0):
    f = plt.figure(figsize=(8, 5))
    if myrange>0:
        plt.plot(range(1, myrange + 1), loss[0:myrange],marker='.')
    else:
        plt.plot(range(1, loss.size(dim=0)+1),loss,marker='.')
    plt.show()
    file_name = path + "/loss.pdf"
    f.savefig(file_name)
    plt.close()



def addErrorGivenWeight(n:int, w:int, batch_size:int = 1):
    errorx = torch.zeros((batch_size, n))
    errorz = torch.zeros((batch_size, n))
    li = list(range(0,n))
    for b in range(batch_size):
        pos = random.sample(li, w) # ランダムに w 個の要素を選択 (重複なし)
        al = torch.rand([w,]) # サイズ w の [0, 1] の乱数
        # それぞれ 1/3 の確率で X エラー, Y エラー, Z エラー
        for p,a in zip(pos,al):
            if a<1/3:
                errorx[b,p] = 1
            elif a<2/3:
                errorz[b,p] = 1
            else:
                errorx[b,p] = 1
                errorz[b,p] = 1
    return errorx, errorz

# %%
# give parameters for the code and decoder
n = 46
k = 2
m = 800 #number of checks, can also use 46 or 44
m1 = m // 2
m2 = m // 2
n_iterations = 6
codeType = 'GB'

# give parameters for training
#learning rate
lr = 0.001
#training for fixed epsilon_0
ep0 = 0.1
#train on errors of weight ranging from r1 to r2
r1_earlier = 2
r2_earlier = 3
# number of updates
n_batches_earlier = 10
#number of error patterns in each mini batch
batch_size = 100

#initialize the decoder, all weights are set to 1
decoder = NBP_2_oc(n, k, m, m1,m2, codeType, n_iterations, batch_size=batch_size, folder_weights=None)
f = plt.figure(figsize=(5, 8))
plt.spy(decoder.H[0].detach().cpu().numpy(), markersize=1, aspect='auto')
plt.title("check matrix of the [["+str(n)+","+str(k)+"]] code with "+str(m)+" checks")
plt.show()

#for comparision, also plot the original check matrix
decoder_2 = NBP_2_oc(n, k, n-k, m1,m2, codeType, n_iterations, batch_size=batch_size, folder_weights=None)
f = plt.figure(figsize=(5, 3))
plt.spy(decoder_2.H[0].detach().cpu().numpy(), markersize=1, aspect='auto')
plt.title("check matrix of the [["+str(n)+","+str(k)+"]] code with "+str(n-k)+" checks")
plt.show()

# path where the training weights are stored, also supports training with previously stored weights
path = "./training_results/" + codeType + "_" + str(n) + "_" + str(k) + "_" + str(m) + "_" + decoder.name + "/"

# ディレクトリが存在しない場合、作成する
if not os.path.exists(path):
    os.makedirs(path)
# %%
