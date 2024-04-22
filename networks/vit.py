import torch.nn as nn
import torch
from LinearRotatoryAttention import LinearRotatoryAttentionModule


class PatchEmbedding(nn.Module):
    def __init__(self, inc: int = 1, patch_size: int = 16, embedding_dim: int = 256):
        super().__init__()
        self.patch_size = patch_size
        self.patcher = nn.Conv2d(in_channels=inc, out_channels=embedding_dim,
                                 kernel_size=patch_size, stride=patch_size, padding=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)

    def forward(self, x):
        image_resolution = x.shape[-1]
        assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image_shape: {image_resolution}, patch size: {self.patch_size}"

        x = self.patcher(x)
        x = self.flatten(x)

        return x.permute(0, 2, 1)


class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_heads: int = 16, attn_dropout: int = 0):
        super().__init__()

        # layer norm
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # multihead attention layer
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim, num_heads=num_heads, dropout=attn_dropout, batch_first=True)

    def forward(self, x):
        x = self.layer_norm(x)
        attn_output, _ = self.multihead_attention(
            query=x, key=x, value=x, need_weights=False)

        return attn_output


class MLPBLock(nn.Module):
    def __init__(self, embedding_dim: int = 256, mlp_size: int = 3072, dropout: int = 0.1):
        super().__init__()

        # create norm layer
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        # create mlp
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim, out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size, out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)

        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_dim: int = 256, num_heads: int = 16, attn_dropout: int = 0,  mlp_size: int = 3072, mlp_dropout: int = 0.1):
        super().__init__()

        # multihead self attention
        self.msa_block = MultiheadSelfAttentionBlock(
            embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout)

        # mlp block
        self.mlp_block = MLPBLock(
            embedding_dim=embedding_dim, mlp_size=mlp_size, dropout=mlp_dropout)

    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, img_size: int = 224,
                 inc: int = 1,
                 patch_size: int = 16,
                 num_transformer_layers: int = 12,
                 embedding_dim: int = 256,
                 mlp_size: int = 3072,
                 num_heads: int = 16,
                 attn_dropout: int = 0,
                 mlp_dropout: int = 0.1,
                 embedding_dropout: int = 0.1,
                 num_classes: int = 8):

        super().__init__()

        assert img_size % patch_size == 0, f"Image size must be divisble by patch size."

        # calculate number of pathces:
        self.num_patches = (img_size * img_size) // (patch_size ** 2)

        # create learnable class embeddings
        self.class_embedding = nn.Parameter(
            data=torch.randn(1, 1, embedding_dim), requires_grad=True)

        # create learnable position embeddings
        self.position_embedding = nn.Parameter(
            data=torch.randn(1, self.num_patches + 1, embedding_dim))

        # create embeddings dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        # patch embedding layer
        self.patch_embedding = PatchEmbedding(
            inc=inc, patch_size=patch_size, embedding_dim=embedding_dim)

        # rotatory attention
        self.rag = LinearRotatoryAttentionModule(
            nft=self.num_patches + 1, dft=embedding_dim,
            nfl=self.num_patches + 1, dfl=embedding_dim,
            nfr=self.num_patches + 1, dfr=embedding_dim,
            dkl=embedding_dim, dvl=(self.num_patches // 4) + 1,
            dkr=embedding_dim, dvr=self.num_patches // 4,
            dkt=embedding_dim, dvt=self.num_patches // 4
        )

        # create transformer encoder block
        self.transformer_encoder = nn.Sequential(
            *[TransformerEncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, attn_dropout=attn_dropout,  mlp_size=mlp_size, mlp_dropout=mlp_dropout) for _ in range(num_transformer_layers)]
        )

        # create classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim, out_features=num_classes)
        )

    def forward(self, x):
        # batch size
        batch_size = x.shape[0]

        # expand class token embedding to match batch size
        class_token = self.class_embedding.expand(batch_size, -1, -1)

        # create the patch embedding
        x = self.patch_embedding(x)  # -> (batch_size, num_patches, embed_dim)

        # concat class token
        x = torch.cat((class_token, x), dim=1)

        # add positional embedding
        x = self.position_embedding + x

        # apply embedding dropout
        x = self.embedding_dropout(x)

        # pass through rotatory attention
        new_output = torch.empty(size=x.shape).to(x.device)
        new_output[0] = x[0]
        new_output[-1] = x[-1]

        for i in range(1, batch_size - 1, 1):
            output = self.rag(
                x[i - 1],
                x[i],
                x[i + 1]
            )
            output = torch.unsqueeze(output, dim=0)
            new_output[i] = output

        x = new_output

        # pass position and patch embedding to transformer encoder
        x = self.transformer_encoder(x)

        # put 0th index logit through classifier
        x = self.classifier(x[:, 0])

        return x


if __name__ == "__main__":
    vit = ViT(img_size=256)

    x = torch.rand(1, 1, 256, 256)
    output = vit(x)
    print(output.shape)
