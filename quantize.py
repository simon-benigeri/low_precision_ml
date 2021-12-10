from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from qtorch import FloatingPoint, FixedPoint


def make_quantizer(config) -> Quantizer:
    Q = None
    if config.quantize:
        # read precision, exp, man, rounding from config
        quantize_config = config.quantize_config
        precision = quantize_config['PRECISION']
        exp = quantize_config[precision]['exp']
        man = quantize_config[precision]['man']
        rounding = quantize_config[precision]['rounding']

        # create low precision FP
        lp_float = FloatingPoint(exp=exp, man=man)

        # create quantizer
        Q = Quantizer(forward_number=lp_float, backward_number=lp_float,
                      forward_rounding=rounding, backward_rounding=rounding)

    return Q