from mindnlp.core import nn, ops

class STFT(nn.Module):
    def __init__(self, filter_length = 256, hop_length = 64):
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.register_buffer('forward_basis_buffer', ops.zeros(258, 1, filter_length)) #TODO: initialize as cos/sin

    def forward(self, input_data):
        input_data0 = input_data.unsqueeze(1)
        to_pad = int((self.filter_length - self.hop_length) / 2)
        input_data1 = nn.functional.pad(ops.unsqueeze(input_data0, 1), (to_pad, to_pad, 0, 0), "reflect")
        forward_transform = nn.functional.conv1d(ops.squeeze(input_data1, 1), self.forward_basis_buffer, None, self.hop_length, 0)
        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        magnitude = ops.sqrt(real_part ** 2 + imag_part ** 2)

        return magnitude

class AdaptiveAudioNormalizationNew(nn.Module):
    def __init__(self, to_pad = 3):
        super().__init__()
        self.to_pad = to_pad
        self.filter_ = nn.Parameter(ops.zeros(1, 1, 2 * to_pad + 1))

    def simple_pad(self, _mean_1, _to_pad_1):
        _left_pad_1  = ops.flip(_mean_1[::1, ::1,  1 : _to_pad_1 + 1 : 1], [-1])             
        _right_pad_1 = ops.flip(_mean_1[::1, ::1, -1 - _to_pad_1: -1 : 1], [-1])            
        return ops.cat([_left_pad_1, _mean_1, _right_pad_1], 2)

    def forward(self, spect):
        spect0 = ops.log1p(spect * 1048576)
        spect1 = ops.unsqueeze(spect0, 0) if spect0.ndim == 2 else spect0
        mean0 = self.simple_pad(ops.mean(spect1, [1], True), self.to_pad)
        mean1 = nn.functional.conv1d(mean0, self.filter_)
        mean_mean = ops.mean(mean1, [-1], True)
        return spect1 + (-mean_mean)


class ConvBlock(nn.Module):
    def __init__(self, in_channels = 258, out_channels = 16, proj = False):
        super().__init__()
        self.dw_conv = nn.Sequential(nn.Conv1d(in_channels, in_channels, 5, padding = 2, groups = in_channels), nn.Identity(), nn.ReLU())
        self.pw_conv = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1), nn.Identity())
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if proj else nn.Identity()
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = self.proj(x)
        x0 = self.pw_conv(self.dw_conv(x))
        x0 += residual
        return self.activation(x0)

class VADDecoderRNNJIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(64, 64, num_layers = 2, batch_first = True, dropout = 0.1)
        self.decoder = nn.Sequential(nn.ReLU(), nn.Conv1d(64, 1, 1), nn.Sigmoid())

    def forward(self, x, hx):
        x, (h, c) = self.rnn(ops.permute(x, (0, 2, 1)), hx)
        return (self.decoder(ops.permute(x, (0, 2, 1))), h, c)


class VADRNNJIT(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = STFT()
        self.adaptive_normalization = AdaptiveAudioNormalizationNew()
        self.first_layer = nn.Sequential(ConvBlock(258, 16, proj = True), nn.Dropout(0.15))
        self.encoder = nn.Sequential(nn.Conv1d(16, 16, 1, stride = 2),
                                     nn.BatchNorm1d(16), nn.ReLU(),
                                     nn.Sequential(ConvBlock(16, 32, proj = True),
                                                   nn.Dropout(0.15)),
                                     nn.Conv1d(32, 32, 1, stride = 2),
                                     nn.BatchNorm1d(32), nn.ReLU(),
                                     nn.Sequential(ConvBlock(32, 32, proj = False),
                                                   nn.Dropout(0.15)),
                                     nn.Conv1d(32, 32, 1, stride = 2),
                                     nn.BatchNorm1d(32), nn.ReLU(),
                                     nn.Sequential(ConvBlock(32, 64, proj = True),
                                                   nn.Dropout(0.15)),
                                     nn.Conv1d(64, 64, 1, stride = 1),
                                     nn.BatchNorm1d(64), nn.ReLU())
        self.decoder = VADDecoderRNNJIT()

    def forward(self, x, hx):
        x0 = self.feature_extractor(x)
        norm = self.adaptive_normalization(x0)
        x1 = ops.cat([x0, norm], 1)
        x2 = self.first_layer(x1)
        x3 = self.encoder(x2)
        if None in hx:
            hx = None
        x4, h0, c0, = self.decoder(x3, hx)
        out = ops.unsqueeze(ops.mean(ops.squeeze(x4, 1), [1]), 1)
        return (out, h0, c0)


class VADRNNJITMerge(nn.Module):
    def __init__(self):
        super().__init__()
        self._model = VADRNNJIT()
        self._model_8k = VADRNNJIT()
        self.sample_rates = [8000, 16000]
        self._last_batch_size = None
        self._last_sr = None
        self._h = None
        self._c = None

        self.reset_states()
    
    def reset_states(self, batch_size = 1):
        self._h = None
        self._c = None
        self._last_sr = 0
        self._last_batch_size = 0

    def _validate_input(self, x, sr):
        # x1 = ops.unsqueeze(x, 0) if x.ndim == 1 else x
        # assert x1.ndim == 2, f"Too many dimensions for input audio chunk {x1.ndim}"
        # sr1, x2 = (16000, x1[:, ::sr // 16000]) if sr != 16000 and sr % 16000 == 0  else (sr, x1)
        # assert sr1 in self.sample_rates, f"Supported sampling rates: {self.sample_rates} (or multiply of 16000)"
        # assert sr1 / x2.shape[1] <= 31.25, "Input audio chunk is too short"
        if x.ndim == 1:
            x1 = ops.unsqueeze(x, 0)
        else:
            x1 = x

        if sr != 16000 and sr % 16000 == 0:
            sr1, x2 = (16000, x1[:, ::sr // 16000])
        else:
            sr1, x2 = sr, x1

        return (x2, sr1)

    def forward(self, x, sr):
        x0, sr0, = self._validate_input(x, sr)

        if self._last_sr and self._last_sr != sr0:
            self.reset_states()
        
        if self._last_batch_size and self._last_batch_size != x0.shape[0]:
            self.reset_states()

        assert sr0 == 16000 or sr0 == 8000
        out, self._h, self._c, = (self._model_8k if sr == 8000 else self._model)(x0, (self._h, self._c))

        self._last_sr = sr0
        self._last_batch_size = self._h.shape[1]
        return out

    def audio_forward(self, x, sr, num_samples: int = 512):
        x, sr = self._validate_input(x, sr)

        if x.shape[1] % num_samples:
            pad_num = num_samples - (x.shape[1] % num_samples)
            x = nn.functional.pad(x, (0, pad_num), 'constant', value=0.0)

        self.reset_states(x.shape[0])
        outs = [self(x[:, i:i+num_samples], sr) for i in range(0, x.shape[1], num_samples)]
        outputs = ops.concat(outs, 1)

        return outputs
