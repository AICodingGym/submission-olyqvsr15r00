"""Training-only control-pattern statistics and within-breath input features."""
import numpy as np
import pandas as pd


class ControlPatterns:
    keys = ['R', 'C', 'position', 'rounded_input']

    @staticmethod
    def frame(df):
        out = df[['R', 'C', 'u_in']].copy()
        out['position'] = df.groupby('breath_id', sort=False).cumcount().to_numpy()
        out['rounded_input'] = np.rint(out.u_in).astype(np.int32)
        return out

    def fit(self, df):
        source = self.frame(df)
        self.table = source.groupby(self.keys).u_in.agg(['mean', 'min', 'max', 'count'])
        self.fallback = source.groupby(self.keys[:3]).u_in.agg(['mean', 'min', 'max'])
        self.global_values = source.u_in.agg(['mean', 'min', 'max'])
        self.training_breath_ids = np.unique(df.breath_id)
        return self

    def transform(self, df):
        source = self.frame(df)
        values = self.table.reindex(pd.MultiIndex.from_frame(source[self.keys])).reset_index(drop=True)
        fallback = self.fallback.reindex(pd.MultiIndex.from_frame(source[self.keys[:3]])).reset_index(drop=True)
        missing = values['count'].isna().to_numpy()
        result = {'pattern_log_count': np.log1p(values['count'].fillna(0).to_numpy()),
                  'pattern_unseen': missing.astype(np.float32)}
        for stat in ['mean', 'min', 'max']:
            value = values[stat].fillna(fallback[stat]).fillna(self.global_values[stat]).to_numpy()
            result['pattern_' + stat] = value
            result['pattern_deviation_' + stat] = source.u_in.to_numpy() - value
        return pd.DataFrame(result, index=df.index).astype(np.float32)


def breath_features(df):
    """All shifts and summaries are confined to the observed 80-row breath."""
    g = df.groupby('breath_id', sort=False)
    out = {}
    for lag in [1, 2, 3]:
        out[f'input_difference_{lag}'] = g.u_in.diff(lag).fillna(0)
        for name in ['u_in', 'u_out', 'time_step']:
            out[f'{name}_future_{lag}'] = g[name].shift(-lag).fillna(0)
    dt = g.time_step.diff().fillna(0)
    out['time_delta'] = dt
    area = (df.u_in * dt).groupby(df.breath_id, sort=False).cumsum()
    out['input_integral'] = area
    out['integral_over_C'] = area / df.C
    out['input_times_R'] = df.u_in * df.R
    out['input_times_C'] = df.u_in * df.C
    out['input_phase_interaction'] = df.u_in * df.u_out
    for stat in ['mean', 'std', 'min', 'max', 'sum']:
        out['breath_input_' + stat] = g.u_in.transform(stat).fillna(0)
    out['input_minus_breath_mean'] = df.u_in - out['breath_input_mean']
    out['inspiration_length'] = (df.u_out == 0).groupby(df.breath_id, sort=False).transform('sum')
    for r in [5, 20, 50]:
        for c in [10, 20, 50]:
            out[f'lung_{r}_{c}'] = ((df.R == r) & (df.C == c)).astype(np.float32)
    return pd.DataFrame(out, index=df.index).astype(np.float32)
