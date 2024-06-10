import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRotatoryAttentionModule(nn.Module):
    # what if i don't need to linearly transform the shit, and all of them are the same
    # key and value matrices are the same
    def __init__(self, seq_length, embed_dim, window_size, value_dim=None, average=True):
        super().__init__()

        if value_dim == None:
            value_dim = embed_dim
        else:
            self.simplre_representation_weight = nn.Parameter(
                torch.randn(embed_dim, value_dim))

        self.average = average
        self.value_dim = value_dim
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.window_size = window_size

        # key weights
        self.key_weights = [nn.Parameter(torch.randn(
            embed_dim, value_dim), requires_grad=True) for _ in range(2 * (window_size - 1))]

        # value weights
        self.value_weights = [nn.Parameter(torch.randn(
            embed_dim, value_dim), requires_grad=True) for _ in range(2 * (window_size - 1))]

        # target key weight
        self.target_key_weight = nn.Parameter(
            torch.randn(embed_dim, value_dim), requires_grad=True)

        # target value weight
        self.target_value_weight = nn.Parameter(
            torch.randn(embed_dim, value_dim), requires_grad=True)

        # biases for attention score
        self.biases = [nn.Parameter(torch.rand(
            1), requires_grad=True) for _ in range(2 * (window_size - 1))]

        # weight for transforming to shape
        self.context_weight = nn.Parameter(torch.randn(seq_length, 1))

    def calculate_attention_score(self, query, key, bias):
        query = torch.transpose(query, 0, 1)

        E = torch.matmul(key, query) + bias
        E = F.tanh(E)

        return E

    def align_attention_score(self, attention_score, value):
        A = F.softmax(attention_score, dim=0)
        A = torch.transpose(A, 0, 1)

        C = torch.matmul(A, value) / torch.sum(A)

        return C

    def context_transformation(self, context_vector):
        # context vector (1, y)
        context_vector = torch.unsqueeze(context_vector, dim=0)

        return torch.matmul(self.context_weight, context_vector)

    def convert_to_correct_device(self, example):
        device = example.device

        for i in range(len(self.key_weights)):
            self.key_weights[i] = self.key_weights[i].to(device)

        for i in range(len(self.value_weights)):
            self.value_weights[i] = self.value_weights[i].to(device)

        for i in range(len(self.biases)):
            self.biases[i] = self.biases[i].to(device)

    def forward(self, Ft, F_list: list):
        """
        Ft: current slice representation
        F: list of other slices
        """
        # calculate simple representation
        # (x, y) -> (1, y) ->(1, y')

        self.convert_to_correct_device(Ft)

        rt = torch.unsqueeze(torch.mean(Ft, dim=0), dim=0)

        if self.value_dim != self.embed_dim:
            rt = torch.matmul(rt, self.simplre_representation_weight)

        assert len(F_list) == self.window_size - \
            1, print(
                f"Error, length of representation lists does not match window size")

        # target key
        Kt = torch.matmul(Ft, self.target_key_weight)

        # target value
        Vt = torch.matmul(Ft, self.target_value_weight)

        # keys
        KEYS = []
        for i, matrix in enumerate(F_list):
            KEYS.append(torch.matmul(matrix, self.key_weights[i]))

        # values
        VALUES = []
        for i, matrix in enumerate(F_list):
            VALUES.append(torch.matmul(matrix, self.value_weights[i]))

        # calculate attention scores based on simple representation
        E = []
        for i, key in enumerate(KEYS):
            E.append(self.calculate_attention_score(
                query=rt, key=key, bias=self.biases[i]))

        # algin attention scores based on simple representations
        R = []
        for attention_score, value in zip(E, VALUES):
            R.append(self.align_attention_score(
                attention_score=attention_score, value=value))

        # calculate attention scores based on target keys
        Et = []
        for i, context_vector in enumerate(R):
            # i + window_size // 2

            Et.append(self.calculate_attention_score(
                query=context_vector, key=Kt, bias=self.biases[i + self.window_size // 2]))

        Rt = []

        for attention_score in Et:
            Rt.append(self.align_attention_score(
                attention_score=attention_score, value=Vt))

        full_representation = R + Rt

        if self.average:
            R = torch.concat(full_representation, dim=0)
            R = torch.mean(R, dim=0)
            R = self.context_transformation(R)
            return R

        else:
            R = torch.concat(full_representation, dim=1)
            R = torch.squeeze(R, dim=0)
            R = self.context_transformation(R)
            return R


if __name__ == "__main__":

    seq_length = 128
    embed_dim = 256
    window_size = 5
    value_dim = embed_dim // (2 * (window_size - 1))
    device = torch.device("cuda")

    # model = DynamicRotatoryAttentionModule(
    #     seq_length=seq_length, embed_dim=embed_dim, window_size=window_size, value_dim=value_dim, average=False)

    model = DynamicRotatoryAttentionModule(
        seq_length=seq_length, embed_dim=embed_dim, window_size=window_size).to(device)

    a = [torch.rand(seq_length, embed_dim).to(device)
         for _ in range(window_size - 1)]

    Ft = torch.randn(seq_length, embed_dim).to(device)

    output = model(Ft, a)

    print(f"Output: {output.shape}")
