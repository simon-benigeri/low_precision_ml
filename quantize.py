from typing import Optional
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
from qtorch import FloatingPoint, FixedPoint


def make_quantizer(config) -> Optional[Quantizer]:
    Q = None

    precision = config.precision
    if precision == 'float32':
        return Q

    # read precision, exp, man, rounding from config
    # precision = quantize_args['precision']
    exp = config.exp
    man = config.man
    rounding = config.rounding

    # create low precision FP
    lp_float = FloatingPoint(exp=exp, man=man)

    # create quantizer
    Q = Quantizer(forward_number=lp_float, backward_number=lp_float,
                  forward_rounding=rounding, backward_rounding=rounding)

    return Q