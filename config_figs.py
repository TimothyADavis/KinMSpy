import matplotlib

matplotlib.rcParams['font.family'] = 'Latin Modern Roman'
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['legend.fontsize'] = 17.5
matplotlib.rcParams['axes.linewidth'] = 1.5
matplotlib.rcParams['xtick.labelsize'] = 20; matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['xtick.major.size'] = 10; matplotlib.rcParams['ytick.major.size'] = 10
matplotlib.rcParams['xtick.major.width'] = 2; matplotlib.rcParams['ytick.major.width'] = 2
matplotlib.rcParams['xtick.minor.size'] = 5; matplotlib.rcParams['ytick.minor.size'] = 5
matplotlib.rcParams['xtick.minor.width'] = 1; matplotlib.rcParams['ytick.minor.width'] = 1
matplotlib.rcParams['xtick.direction'] = 'in'; matplotlib.rcParams['ytick.direction'] = 'in'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['ytick.left'] = True
params = {'mathtext.default': 'regular' }
matplotlib.rcParams.update(params)
matplotlib.rcParams['axes.labelsize'] = 30