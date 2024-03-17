class ParametricSequential(nn.Sequential):
    """
    A sequential container for modules with parameters.
    Modules will be executed in the order they are added.
    """
    def __init__(self, *args):
        super(ParametricSequential, self).__init__(*args)

    def forward(self, x, **kwargs):
        for module in self._modules.values():
            x = module(x, **kwargs)
        return x

class DropConvBlock(nn.Module):
    """
    Convolution block with Batch normalization, ReLU activation, and Dropout layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 bias=False,
                 dropout_prob=0.0):
        super(DropConvBlock, self).__init__()
        self.use_dropout = (dropout_prob != 0.0)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activ = nn.ReLU(inplace=True)
        if self.use_dropout:
            self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


def drop_conv3x3_block(in_channels,
                       out_channels,
                       stride=1,
                       padding=1,
                       bias=False,
                       dropout_prob=0.0):
    """
    3x3 version of the convolution block with dropout.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    return DropConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        dropout_prob=dropout_prob)


class FractalBlock(nn.Module):
    """
    FractalNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_columns : int
        Number of columns in each block.
    loc_drop_prob : float
        Local drop path probability.
    dropout_prob : float
        Probability of dropout.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_columns,
                 loc_drop_prob,
                 dropout_prob):
        super(FractalBlock, self).__init__()
        assert (num_columns >= 1)
        self.num_columns = num_columns
        self.loc_drop_prob = loc_drop_prob

        self.blocks = nn.Sequential()
        depth = 2 ** (num_columns - 1)
        for i in range(depth):
            level_block_i = nn.Sequential()
            for j in range(self.num_columns):
                column_step_j = 2 ** j
                if (i + 1) % column_step_j == 0:
                    in_channels_ij = in_channels if (i + 1 == column_step_j) else out_channels
                    level_block_i.add_module("subblock{}".format(j + 1), drop_conv3x3_block(
                        in_channels=in_channels_ij,
                        out_channels=out_channels,
                        dropout_prob=dropout_prob))
            self.blocks.add_module("block{}".format(i + 1), level_block_i)

    @staticmethod
    def calc_drop_mask(batch_size,
                       glob_num_columns,
                       curr_num_columns,
                       max_num_columns,
                       loc_drop_prob):
        """
        Calculate drop path mask.

        Parameters:
        ----------
        batch_size : int
            Size of batch.
        glob_num_columns : int
            Number of columns in global drop path mask.
        curr_num_columns : int
            Number of active columns in the current level of block.
        max_num_columns : int
            Number of columns for all network.
        loc_drop_prob : float
            Local drop path probability.

        Returns
        -------
        Tensor
            Resulted mask.
        """
        glob_batch_size = glob_num_columns.shape[0]
        glob_drop_mask = np.zeros((curr_num_columns, glob_batch_size), dtype=np.float32)
        glob_drop_num_columns = glob_num_columns - (max_num_columns - curr_num_columns)
        glob_drop_indices = np.where(glob_drop_num_columns >= 0)[0]
        glob_drop_mask[glob_drop_num_columns[glob_drop_indices], glob_drop_indices] = 1.0

        loc_batch_size = batch_size - glob_batch_size
        loc_drop_mask = np.random.binomial(
            n=1,
            p=(1.0 - loc_drop_prob),
            size=(curr_num_columns, loc_batch_size)).astype(np.float32)
        alive_count = loc_drop_mask.sum(axis=0)
        dead_indices = np.where(alive_count == 0.0)[0]
        loc_drop_mask[np.random.randint(0, curr_num_columns, size=dead_indices.shape), dead_indices] = 1.0

        drop_mask = np.concatenate((glob_drop_mask, loc_drop_mask), axis=1)
        return torch.from_numpy(drop_mask)

    @staticmethod
    def join_outs(raw_outs,
                  glob_num_columns,
                  num_columns,
                  loc_drop_prob,
                  training):
        """
        Join outputs for current level of block.

        Parameters:
        ----------
        raw_outs : list of Tensor
            Current outputs from active columns.
        glob_num_columns : int
            Number of columns in global drop path mask.
        num_columns : int
            Number of columns for all network.
        loc_drop_prob : float
            Local drop path probability.
        training : bool
            Whether training mode for network.

        Returns
        -------
        Tensor
            Joined output.
        """
        curr_num_columns = len(raw_outs)
        out = torch.stack(raw_outs, dim=0)
        assert (out.size(0) == curr_num_columns)

        if training:
            batch_size = out.size(1)
            batch_mask = FractalBlock.calc_drop_mask(
                batch_size=batch_size,
                glob_num_columns=glob_num_columns,
                curr_num_columns=curr_num_columns,
                max_num_columns=num_columns,
                loc_drop_prob=loc_drop_prob)
            batch_mask = batch_mask.to(out.device)
            assert (batch_mask.size(0) == curr_num_columns)
            assert (batch_mask.size(1) == batch_size)
            batch_mask = batch_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            masked_out = out * batch_mask
            num_alive = batch_mask.sum(dim=0)
            num_alive[num_alive == 0.0] = 1.0
            out = masked_out.sum(dim=0) / num_alive
        else:
            out = out.mean(dim=0)

        return out

    def forward(self, x, glob_num_columns):
        outs = [x] * self.num_columns

        for level_block_i in self.blocks._modules.values():
            outs_i = []

            for j, block_ij in enumerate(level_block_i._modules.values()):
                input_i = outs[j]
                outs_i.append(block_ij(input_i))

            joined_out = FractalBlock.join_outs(
                raw_outs=outs_i[::-1],
                glob_num_columns=glob_num_columns,
                num_columns=self.num_columns,
                loc_drop_prob=self.loc_drop_prob,
                training=self.training)

            len_level_block_i = len(level_block_i._modules.values())
            for j in range(len_level_block_i):
                outs[j] = joined_out

        return outs[0]


class FractalUnit(nn.Module):
    """
    FractalNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    num_columns : int
        Number of columns in each block.
    loc_drop_prob : float
        Local drop path probability.
    dropout_prob : float
        Probability of dropout.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_columns,
                 loc_drop_prob,
                 dropout_prob):
        super(FractalUnit, self).__init__()
        self.block = FractalBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            num_columns=num_columns,
            loc_drop_prob=loc_drop_prob,
            dropout_prob=dropout_prob)
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2)

    def forward(self, x, glob_num_columns):
        x = self.block(x, glob_num_columns=glob_num_columns)
        x = self.pool(x)
        return x
