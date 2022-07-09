"""
Struct that will be serialized and sent to workers for processing
"""
class SubSample:
    def __init__(self,block_pos:tuple,P,Q,b_u,b_i,b,y,
                alpha,beta1,beta2) -> None:
        self.block_pos = block_pos #used to get numpy array of samples in shared samples grid
        self.P = P #this is a subset of P belonging to the users of the block
        self.Q = Q
        self.b_u=b_u
        self.b_i=b_i
        self.b=b
        self.y=y
        self.alpha=alpha
        self.beta1=beta1
        self.beta2=beta2