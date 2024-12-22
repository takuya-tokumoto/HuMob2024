import torch
from torch import nn


class DayEmbeddingModel(nn.Module):
    """
    0: <PAD>
    """

    def __init__(self, embed_size):
        super(DayEmbeddingModel, self).__init__()

        self.day_embedding = nn.Embedding(
            num_embeddings=75 + 1,
            embedding_dim=embed_size,
        )

    def forward(self, day):
        embed = self.day_embedding(day)
        return embed


class TimeEmbeddingModel(nn.Module):
    """
    0: <PAD>
    """

    def __init__(self, embed_size):
        super(TimeEmbeddingModel, self).__init__()

        self.time_embedding = nn.Embedding(
            num_embeddings=48 + 1,
            embedding_dim=embed_size,
        )

    def forward(self, time):
        embed = self.time_embedding(time)
        return embed


class LocationXEmbeddingModel(nn.Module):
    """
    0: <PAD>
    201: <MASK>
    """

    def __init__(self, embed_size):
        super(LocationXEmbeddingModel, self).__init__()

        self.location_embedding = nn.Embedding(
            num_embeddings=202,
            embedding_dim=embed_size,
        )

    def forward(self, location):
        embed = self.location_embedding(location)
        return embed


class LocationYEmbeddingModel(nn.Module):
    """
    0: <PAD>
    201: <MASK>
    """

    def __init__(self, embed_size):
        super(LocationYEmbeddingModel, self).__init__()

        self.location_embedding = nn.Embedding(
            num_embeddings=202,
            embedding_dim=embed_size,
        )

    def forward(self, location):
        embed = self.location_embedding(location)
        return embed


class TimedeltaEmbeddingModel(nn.Module):
    """
    0: <PAD>
    """

    def __init__(self, embed_size):
        super(TimedeltaEmbeddingModel, self).__init__()

        self.timedelta_embedding = nn.Embedding(
            num_embeddings=48,
            embedding_dim=embed_size,
        )

    def forward(self, timedelta):
        embed = self.timedelta_embedding(timedelta)
        return embed


class EmbeddingLayer(nn.Module):
    def __init__(self, embed_size):
        super(EmbeddingLayer, self).__init__()

        self.day_embedding = DayEmbeddingModel(embed_size)
        self.time_embedding = TimeEmbeddingModel(embed_size)
        self.location_x_embedding = LocationXEmbeddingModel(embed_size)
        self.location_y_embedding = LocationYEmbeddingModel(embed_size)
        self.timedelta_embedding = TimedeltaEmbeddingModel(embed_size)

    def forward(self, day, time, location_x, location_y, timedelta):
        day_embed = self.day_embedding(day)
        time_embed = self.time_embedding(time)
        location_x_embed = self.location_x_embedding(location_x)
        location_y_embed = self.location_y_embedding(location_y)
        timedelta_embed = self.timedelta_embedding(timedelta)

        embed = day_embed + time_embed + location_x_embed + location_y_embed + timedelta_embed
        return embed


class TransformerEncoderModel(nn.Module):
    def __init__(self, layers_num, heads_num, embed_size):
        super(TransformerEncoderModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads_num)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=self.encoder_layer, num_layers=layers_num)

    def forward(self, input, src_key_padding_mask):
        out = self.transformer_encoder(input, src_key_padding_mask=src_key_padding_mask)
        return out


class POIConvNet(nn.Module):
    def __init__(self, embed_size):
        super(POIConvNet, self).__init__()

        # POIデータの畳み込み層を定義 (2D Convolution)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, embed_size, kernel_size=3, padding=1)

        # Global Poolingで(200, 200)を(128,)のベクトルに
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, poi_tensor):
        # 畳み込み処理を行う
        x = F.relu(self.conv1(poi_tensor))  # (Batch Size, 32, 200, 200)
        x = F.relu(self.conv2(x))  # (Batch Size, 64, 200, 200)
        x = F.relu(self.conv3(x))  # (Batch Size, 128, 200, 200)

        # Global Poolingで(128, 1, 1)に縮約し、(128,)のベクトルに
        x = self.global_pool(x).squeeze(-1).squeeze(-1)  # (Batch Size, 16)

        return x


def create_poi_tensor(poi_path, grid_size=200):

    # CSV.GZファイルを読み込む
    poi_data = pl.read_csv(poi_path)

    # 200 x 200 のグリッドを初期化 (初期値はすべて0)
    poi_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    # POIデータを座標ごとにグリッドに加算
    for row in poi_data.iter_rows():
        x, y, poi_count = row[0] - 1, row[1] - 1, row[3]  # x, y, POI_count に対応
        poi_grid[x, y] += poi_count  # 座標に基づいてPOIの数を加算

    poi_tensor = torch.tensor(poi_grid).unsqueeze(0)  # (1, 200, 200)
    poi_tensor = (poi_tensor - poi_tensor.min()) / (poi_tensor.max() - poi_tensor.min())

    return poi_tensor


class FFNLayer(nn.Module):
    def __init__(self, embed_size):
        super(FFNLayer, self).__init__()

        # POI ConvNet (200, 200) -> (16) を行う畳み込みネットワーク
        self.poi_conv_net = POIConvNet()

        self.ffn1 = nn.Sequential(
            nn.Linear(embed_size * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )
        self.ffn2 = nn.Sequential(
            nn.Linear(embed_size * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 200),
        )

    def forward(self, input):

        # POIデータを畳み込み処理して16次元に
        poi_path = "/kaggle/s3storage/01_public/humob-challenge-2024/input/POIdata_cityB.csv"
        poi_tensor = create_poi_tensor(poi_path, grid_size=200)
        poi_features = self.poi_conv_net(poi_tensor)

        input = torch.cat([input, poi_features], dim=-1)

        output_x = self.ffn1(input)
        output_y = self.ffn2(input)

        output = torch.stack([output_x, output_y], dim=-2)
        return output


class LPBERT(nn.Module):
    def __init__(self, layers_num, heads_num, embed_size):
        super(LPBERT, self).__init__()

        self.embedding_layer = EmbeddingLayer(embed_size)
        self.transformer_encoder = TransformerEncoderModel(layers_num, heads_num, embed_size)
        self.ffn_layer = FFNLayer(embed_size)

    def forward(self, day, time, location_x, location_y, timedelta, len):
        embed = self.embedding_layer(day, time, location_x, location_y, timedelta)
        embed = embed.transpose(0, 1)

        max_len = day.shape[-1]
        indices = torch.arange(max_len, device=len.device).unsqueeze(0)
        src_key_padding_mask = ~(indices < len.unsqueeze(-1))

        transformer_encoder_output = self.transformer_encoder(embed, src_key_padding_mask)
        transformer_encoder_output = transformer_encoder_output.transpose(0, 1)

        output = self.ffn_layer(transformer_encoder_output)
        return output
