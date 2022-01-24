import torch.nn as nn
import torch
# import torch.nn.functional as F
from basic_blocks import ResnetBlock, ResnetBlock_ds, DRN_block
# import functools
# from torch.optim import lr_scheduler

# import Function
# The height of each image.
IMG_HEIGHT = 256

# The width of each image.
IMG_WIDTH = 256

ngf = 32

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, ngf=32, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='zero', no_antialias=True, opt=None):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt

        use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer == nn.InstanceNorm2d:
            eps = 1e-6
        else:
            eps = 0.001
        padding_layer = nn.ZeroPad2d # cyclegan used reflection

        model = [padding_layer(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf, affine=True, eps=eps),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if(no_antialias):
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2, affine=True, eps=eps),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2, affine=True, eps=eps),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)]
                          
        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, is_GA=True)]
    
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                          norm_layer(int(ngf * mult / 2), affine=True, eps=eps),
                          nn.ReLU(True)]
            else:
                model += [Upsample(ngf * mult),
                          nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                    kernel_size=3, stride=1,
                                    padding=1,  # output_padding=1,
                                    bias=use_bias),
                          norm_layer(int(ngf * mult / 2), affine=True, eps=eps),
                          nn.ReLU(True)]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding='same')]

        self.model = nn.Sequential(*model)
        self.activation = nn.Tanh()

    def forward(self, input, layers=[], encode_only=False, debug=False):

        if debug:
            return self.debug(input)

        if -1 in layers:
            layers.append(len(self.model))
        if len(layers) > 0:
            feat = input
            feats = []
            for layer_id, layer in enumerate(self.model):
                # print(layer_id, layer.__class__.__name__)
                feat = layer(feat)
                if layer_id in layers:
                    # print("%d: adding the output of %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    feats.append(feat)
                else:
                    # print("%d: skipping %s %d" % (layer_id, layer.__class__.__name__, feat.size(1)))
                    pass
                if layer_id == layers[-1] and encode_only:
                    # print('encoder only return features')
                    return feats  # return intermediate features alone; stop in the last layers

            return feat, feats  # return both output and intermediate features
        else:
            """Standard forward"""
            fake = self.model(input)
            out = self.activation(fake+input)
            return out

# gen = ResnetGenerator()
# input = torch.randn(8,1,256,256)
# print(gen(input).size())


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, no_antialias=True):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        use_bias = False
        if norm_layer == nn.InstanceNorm2d:
            eps = 1e-6
        else:
            eps = 0.001

        kw = 4
        padw = 2     #padw=2 gives 35*35 padw=1 gives 30*30
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias = use_bias), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw, bias = use_bias), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        # self.conv = nn.Sequential(*sequence)

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult, affine=True, eps=eps),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult, affine=True, eps=eps),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult, affine=True, eps=eps),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias = use_bias)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)
        self.debug = nn.Sequential(*sequence)

    def forward(self, input, debug=False):
        """Standard forward."""
        if debug:
            print("debug")
            return self.debug(input)
        return self.model(input)

class NLayerDiscriminator_aux(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc=1, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, no_antialias=True):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator_aux, self).__init__()
        
        use_bias = False  #norm_layer == nn.InstanceNorm2d
        if norm_layer == nn.InstanceNorm2d:
            eps = 1e-6
        else:
            eps = 0.001

        kw = 4
        padw = 2     #padw=2 gives 35*35 padw=1 gives 30*30
        if(no_antialias):
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, bias = use_bias), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True), Downsample(ndf)]
        # self.conv = nn.Sequential(*sequence)

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult, affine=True, eps=eps),
                    nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult, affine=True, eps=eps),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult, affine=True, eps=eps),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 2, kernel_size=kw, stride=1, padding=padw, bias=use_bias)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)[:,0,:,:].unsqueeze(1), self.model(input)[:,1,:,:].unsqueeze(1),

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc=1, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        use_bias = False  #norm_layer == nn.InstanceNorm2d
        if norm_layer == nn.InstanceNorm2d:
            eps = 1e-6
        else:
            eps = 0.001

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2, affine=True, eps=eps),
            nn.LeakyReLU(0.2, True),
            # nn.Conv2d(ndf*2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=use_bias),
            # norm_layer(ndf * 4, affine=True, eps=eps),
            # nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)

class PixelDiscriminator_aux(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc=1, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator_aux, self).__init__()
        use_bias = False  #norm_layer == nn.InstanceNorm2d
        if norm_layer == nn.InstanceNorm2d:
            eps = 1e-6
        else:
            eps = 0.001

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2, affine=True, eps=eps),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 2, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        # self.model(input)[:,0,:,:].unsqueeze(1), self.model(input)[:,1,:,:].unsqueeze(1),
        return self.net(input)[:,0,:,:].unsqueeze(1), self.net(input)[:,1,:,:].unsqueeze(1)

class segmenter(nn.Module):
    def __init__(self, channel=1, out_channel=512, norm_layer = nn.BatchNorm2d):
        super(segmenter, self).__init__()
        fb = 16
        k1 = 3
        padding = "zero"
        use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer == nn.InstanceNorm2d:
            eps = 1e-6
        else:
            eps = 0.001

        self.pad1 = nn.ZeroPad2d(3) # the same padding
        self.conv1 = nn.Sequential(nn.Conv2d(channel, fb, 7, bias=False), norm_layer(fb, eps=eps), nn.ReLU())
        self.res1 = ResnetBlock(fb, padding_type=padding, norm_layer = norm_layer, use_dropout=False)
        self.maxpool1 = nn.MaxPool2d(2, 2)

        #out1 = tf.nn.max_pool(o_r1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.res_ds1 = ResnetBlock_ds(fb*2, padding_type=padding, norm_layer = norm_layer, use_dropout=False)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        #out2 = maxpool(out2) # im not gonna do the padding is same because it is absolutely garbage
        #out2 = tf.nn.max_pool(o_r2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.res_ds2 =ResnetBlock_ds(fb*4, padding_type=padding, norm_layer = norm_layer, use_dropout=False)
        self.res2 = ResnetBlock(fb*4, padding_type=padding, norm_layer = norm_layer, use_dropout=False)
        self.maxpool3 = nn.MaxPool2d(2, 2)

        #out3 = tf.nn.max_pool(o_r4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.res_ds3 = ResnetBlock_ds(fb*8, padding_type=padding, norm_layer = norm_layer, use_dropout=False)
        self.res3 = ResnetBlock(fb*8, padding_type=padding, norm_layer = norm_layer, use_dropout=False)


        self.res_ds4 = ResnetBlock_ds(fb*16, padding_type=padding, norm_layer = norm_layer, use_dropout=False)
        self.res4 = ResnetBlock(fb*16, padding_type=padding, norm_layer = norm_layer, use_dropout=False)

        self.res5 = ResnetBlock(fb*16, padding_type=padding, norm_layer = norm_layer, use_dropout=False)
        self.res6 = ResnetBlock(fb*16, padding_type=padding, norm_layer = norm_layer, use_dropout=False)

        if fb*16 == out_channel:
            self.res_ds5 = ResnetBlock(out_channel, padding_type=padding, norm_layer = norm_layer, use_dropout=False)
        else:
            self.res_ds5 = ResnetBlock_ds(out_channel, padding_type=padding, norm_layer = norm_layer, use_dropout=False)
        self.res7 = ResnetBlock(out_channel, padding_type=padding, norm_layer = norm_layer, use_dropout=False)

        self.drn1 = DRN_block(out_channel, out_channel, norm_layer=norm_layer)
        self.drn2 = DRN_block(out_channel, out_channel, norm_layer=norm_layer)


        self.pad2 = nn.ZeroPad2d(1)
        #conv = general_conv2d(fb*32, fb*32, k1, k1, 1, 1, 0.01, 'SAME', norm_type='Batch', keep_rate=keep_rate)
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, k1, bias=use_bias), norm_layer(out_channel, eps=eps), nn.ReLU())
        self.pad3 = nn.ZeroPad2d(1)
        # conv4 = general_conv2d(fb*32, fb*32, k1, k1, 1, 1, 0.01, 'SAME', norm_type='Batch', keep_rate=keep_rate)
        self.conv3 = nn.Sequential(nn.Conv2d(out_channel, out_channel, k1, bias=use_bias), norm_layer(out_channel, eps=eps), nn.ReLU())

    def forward(self, x):

        out = self.pad1(x)
        out = self.conv1(out)
        out = self.res1(out)
        out = self.maxpool1(out)
        out = self.res_ds1(out)
        out = self.maxpool2(out)
        out = self.res_ds2(out)
        out = self.res2(out)
        out = self.maxpool3(out)
        out = self.res_ds3(out)
        out = self.res3(out)
        out = self.res_ds4(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.res6(out)
        out = self.res_ds5(out)
        out = self.res7(out)
        mid_way = out
        out = self.drn1(out)
        out = self.drn2(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.pad3(out)
        out = self.conv3(out)


        return out, mid_way


class decoder(nn.Module):
    def __init__(self, channel=512, out_channel=1, norm_layer = nn.InstanceNorm2d):
        super(decoder, self).__init__()
        ngf = 32

        n_downsampled = 2
        f = 7
        ks = 3
        padding = 'zero'
        use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer == nn.InstanceNorm2d:
            eps = 1e-6
        else:
            eps = 0.001

        model = []
        model += [nn.ZeroPad2d(1),
                  nn.Sequential(nn.Conv2d(channel, ngf * 4, ks, bias=False), 
                  norm_layer(ngf*4, affine=True, eps=eps), 
                  nn.ReLU())]

        for i in range(4):
            model += [ResnetBlock(ngf * 4, padding_type=padding, norm_layer = norm_layer, use_dropout=False)]

            
        model += [nn.ConvTranspose2d(ngf * 4, int(ngf * 2),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=True),
                  norm_layer(int(ngf * 2), affine=True, eps=eps),
                  nn.ReLU(True)]

        model += [nn.ConvTranspose2d(ngf * 2, int(ngf * 2),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=True),
                  norm_layer(int(ngf * 2), affine=True, eps=eps),
                  nn.ReLU(True)]

        model += [nn.ConvTranspose2d(ngf * 2, int(ngf),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=True),
                  norm_layer(int(ngf), affine=True, eps=eps),
                  nn.ReLU(True)]

        model += [nn.ZeroPad2d(3)]
        model += [nn.Conv2d(ngf, out_channel, kernel_size=f, padding=0, bias=False)]
        # model += [nn.Tanh()]
        self.model = nn.Sequential(*model)
        self.activation = nn.Tanh()

    def forward(self, input, refimg):
        fake = self.model(input)
        out = self.activation(fake+refimg)
        return out

class Build_classifier(nn.Module):
    def __init__(self, channel=512, out = 5):
        super(Build_classifier, self).__init__()

        self.classifier = nn.Conv2d(channel, out, 1, bias=False)
    def forward(self, x):
        self.out = self.classifier(x)

        return  nn.functional.interpolate(self.out, (IMG_HEIGHT, IMG_WIDTH), mode='bilinear')


class Build_bitmask(nn.Module):
    def __init__(self, channel=128, out = 1):
        super(Build_bitmask, self).__init__()

        self.classifier = nn.Conv2d(channel, out, 1, bias=False)


    def forward(self, x):
        # print(x.size())
        self.out = self.classifier(x)
        # print(self.out.size())
        flat = self.out.view(8, -1)

        # print(self.out.size())
        
        medians = torch.median(flat, dim = 1)
        # print(medians)
        for i in range(8):
            # [self.out[i] <= medians[0][i]]
            zeros = self.out[i] <= medians[0][i]
            ones = self.out[i] > medians[0][i]
            self.out[i][zeros] = 0.
            self.out[i][ones] = 1.
        # self.out = self.out()

        return  nn.functional.interpolate(self.out, (IMG_HEIGHT, IMG_WIDTH), mode='area')


import torch
import torch.nn as nn

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, 
                 ConvNd=nn.Conv3d,
                 BatchNormNd=nn.BatchNorm3d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)

        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)

        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))

        return out

class GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_2D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize)

class GloRe_Unit_3D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_3D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv3d,
                                            BatchNormNd=nn.BatchNorm3d,
                                            normalize=normalize)