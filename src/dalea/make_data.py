import numpy as np

from .samplers_basic import sample_categorical


def make_one_node(
        n_observations, activation_fn, target_type='continuous'):

    n_features = 1
    n_targets = 1
    n_hidden_nodes = 1

    beta_0 = np.array(0.8).reshape(n_features, n_hidden_nodes)
    gamma_0 = np.array(-0.1).reshape(1, n_hidden_nodes)
    tau_0 = np.array(0.5).reshape(1, n_hidden_nodes)
    sigma_1 = np.array(0.3).reshape(1, n_hidden_nodes)
    beta_1 = np.array(-2.0).reshape(n_hidden_nodes, n_targets)
    gamma_1 = np.array(0.2).reshape(1, n_targets)
    tau_1 = np.array(0.8).reshape(1, n_targets)
    # if no activation, equivalent to
    # -1.6x + 0.4 + Norm[0, sqrt(0.2)**2]

    n_realizations = 200

    x = np.random.randn(n_observations, n_features)
    u_0 = x
    v_0_mean = u_0 @ beta_0 + gamma_0
    epsilon_0 = np.random.randn(n_realizations, *v_0_mean.shape) * tau_0
    v_0 = v_0_mean + epsilon_0
    u_1_mean = activation_fn(v_0)
    delta_1 = np.random.randn(*u_1_mean.shape) * sigma_1
    u_1 = u_1_mean + delta_1
    v_1_mean = u_1 @ beta_1 + gamma_1
    epsilon_1 = np.random.randn(*v_1_mean.shape) * tau_1
    v_1 = v_1_mean + epsilon_1
    if target_type == 'continuous':
        y = v_1
    elif target_type == 'categorical':
        y = sample_categorical(v_1)

    i_realized = 0
    v_0_realized = v_0[i_realized]
    v_1_realized = v_1[i_realized]
    u_1_realized = u_1[i_realized]
    y_realized = y[i_realized]
    if target_type == 'continuous':
        v_realized = [v_0_realized, 'y']
    elif target_type == 'categorical':
        v_realized = [v_0_realized, v_1_realized]

    params = {
            'beta': [beta_0, beta_1],
            'gamma': [gamma_0, gamma_1],
            'tau': [tau_0, tau_1],
            'sigma': [np.nan, sigma_1],
            'v': v_realized,
            'u': ['x', u_1_realized],
            'y_dist': y}

    return x, y_realized, params


def wave(
        x, period=1.0, center=0.0, amp=1.0,
        base=0.0, slope=0.0):
    u = np.max(x, -1, keepdims=True)  # in case x is multi-dim
    z = (u - center) / period
    y = np.sin(z * 2*np.pi) * amp + base + slope * z
    return y


def sample_bernoulli(mean_func, x, shape=(), seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean = mean_func(x)
    assert 0 <= np.min(mean) <= np.max(mean) <= 1
    y = np.random.binomial(n=1, p=mean, size=shape+mean.shape)
    return y, mean


def sample_normal(mean_func, std_func, x, shape=(), seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean = mean_func(x)
    std = std_func(x)
    assert std.min() >= 0
    y = np.random.normal(
            mean, std, size=shape+mean.shape)
    return y, mean, std


def sample_chisq(mean_func, std_func, x, shape=(), seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean = mean_func(x)
    std = std_func(x)
    assert std.min() >= 0
    df = 1
    noise = np.random.chisquare(
            df=df, size=shape+mean.shape)
    noise = (noise - df) / np.sqrt(df * 2)
    # noise_sign_is_pos = x.mean(-1, keepdims=True) > 0
    # noise_sign = noise_sign_is_pos * 2 - 1
    # noise = noise * noise_sign
    y = mean + noise * std
    return y, mean, std


def sample_multimodal(mean_func, std_func, x, shape=(), seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean = mean_func(x)
    std = std_func(x)
    assert std.min() >= 0
    mode = np.sign(np.random.randn(*shape, *mean.shape))
    ind = x.mean(-1, keepdims=True) > 0
    sep = ind
    scale = 0.1
    noise = np.random.randn(*shape, *mean.shape)
    noise = noise * scale + mode * sep
    noise = noise / noise.std()
    y = mean + noise * std
    return y, mean, std


def sample_mixture(mean_func, std_func, x, shape=(), seed=None):
    if seed is not None:
        np.random.seed(seed)
    mean = mean_func(x)
    std = std_func(x)
    assert std.min() >= 0
    ind = np.random.randn(*shape, *mean.shape) > 0
    sign = np.sign(x.mean(-1, keepdims=True))
    df = 2
    dist_0 = np.random.chisquare(df=df, size=shape+mean.shape)
    dist_1 = np.random.randn(*shape, *mean.shape) * 0.1 - df
    noise = dist_0 * (1 - ind) + dist_1 * ind
    noise = noise * sign
    noise = noise / noise.std()
    y = mean + noise * std
    return y, mean, std


def gen_x(n_observations_train, n_observations_test, dist):
    if dist == 'uniform':
        x_train = np.linspace(-4, 4, n_observations_train).reshape(-1, 1)
    elif dist == 'radial':
        x_train = np.concatenate([
            np.linspace(-4, -2, n_observations_train // 10 * 1),
            np.linspace(-2, -0, n_observations_train // 10 * 4),
            np.linspace(0, 2, n_observations_train // 10 * 4),
            np.linspace(2, 4, n_observations_train // 10 * 1),
            ]).reshape(-1, 1)
    x_test = np.linspace(-4, 4, n_observations_test).reshape(-1, 1)
    return x_train, x_test


def gen_y(x, target_type, noise_type=None, std_type=None, seed=None):
    if target_type == 'continuous':
        if noise_type == 'normal':
            sample_func = sample_normal
        elif noise_type == 'chisq':
            sample_func = sample_chisq
        elif noise_type == 'multimodal':
            sample_func = sample_multimodal
        elif noise_type == 'mixture':
            sample_func = sample_mixture
        else:
            raise ValueError('Distribution not recognized')
    elif target_type == 'categorical':
        sample_func = sample_bernoulli
    else:
        raise ValueError('Target type not recognized')

    if target_type == 'continuous':

        def mean_func(x):
            return wave(x, period=2, amp=1, base=0, slope=2)

        if std_type == 'periodic':
            def std_func(x):
                return (wave(x, period=8, center=-4) < 0) * 0.8 + 0.2
        elif std_type == 'radial':
            def std_func(x):
                return (wave(np.abs(x), period=4, center=-2) < 0) * 0.8 + 0.2
        elif std_type == 'uniform':
            def std_func(x):
                return np.ones_like(x)
        y, y_mean, y_std = sample_func(
                mean_func, std_func, x, seed=seed)

    else:

        def mean_func(x):
            return wave(
                    x, period=2.0, center=0.0, amp=0.5,
                    base=0.5, slope=0.0)

        y, y_mean = sample_func(mean_func, x, seed=seed)
        y_std = None
    return y, y_mean, y_std


def simulate_data(
        n_observations_train, n_observations_test,
        target_type, noise_type='normal', seed=None):
    if noise_type == 'normal':
        dist_x = 'radial'
        std_type = 'periodic'
    elif noise_type == 'chisq':
        dist_x = 'radial'
        std_type = 'periodic'
    elif noise_type == 'multimodal':
        dist_x = 'radial'
        std_type = 'uniform'
    elif noise_type == 'mixture':
        dist_x = 'uniform'
        std_type = 'uniform'
    else:
        raise ValueError('`noise_type` not recognized')
    x_train, x_test = gen_x(n_observations_train, n_observations_test, dist_x)
    y_train, y_mean_train, y_std_train = gen_y(
            x_train, 'continuous', noise_type, std_type, seed=seed)
    y_test, y_mean_test, y_std_test = gen_y(
            x_test, 'continuous', noise_type, std_type, seed=seed)
    return (
            x_train, x_test,
            y_train, y_test,
            y_mean_train, y_mean_test,
            y_std_train, y_std_test)
