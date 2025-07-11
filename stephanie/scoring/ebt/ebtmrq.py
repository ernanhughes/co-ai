class EnergyTunedMRQ:
    def __init__(self, ebt, mrq):
        self.ebt = ebt
        self.mrq = mrq
        self.ebt_refine_threshold = ebt.refine_threshold
        self.ebt_fallback_threshold = ebt.fallback_threshold    