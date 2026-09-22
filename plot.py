import numpy as np
import pickle, os
import matplotlib.pyplot as plt

def avg_return_curve(x, y, stride, total_steps):
    """
    Author: Rupam Mahmood (armahmood@ualberta.ca)
    :param x: A list of list of termination steps for each episode. len(x) == total number of runs
    :param y: A list of list of episodic return. len(y) == total number of runs
    :param stride: The timestep interval between two aggregate datapoints to be calculated
    :param total_steps: The total number of time steps to be considered
    :return: time steps for calculated data points, average returns for each data points, std-errs
    """
    assert len(x) == len(y)
    num_runs = len(x)
    avg_ret = np.zeros(total_steps // stride)
    stderr_ret = np.zeros(total_steps // stride)
    steps = np.arange(stride, total_steps + stride, stride)
    for i in range(0, total_steps // stride):
        rets = []
        avg_rets_per_run = []
        for run in range(num_runs):
            xa = np.array(x[run])
            ya = np.array(y[run])
            rets.append(ya[np.logical_and(i * stride < xa, xa <= (i + 1) * stride)].tolist())
            avg_rets_per_run.append(np.mean(rets[-1]))
        avg_ret[i] = np.mean(avg_rets_per_run)
        stderr_ret[i] = np.std(avg_rets_per_run) / np.sqrt(num_runs)
    return steps, avg_ret, stderr_ret

def main(data_dir, int_space, total_steps):
    plt.figure(figsize=(8, 5))
    all_termination_time_steps, all_episodic_returns, env_name = [], [], ''
    for file in os.listdir(data_dir):
        if file.endswith(".pkl"):
            with open(os.path.join(data_dir, file), "rb") as f:
                episodic_returns, termination_time_steps, env_name = pickle.load(f)
                all_termination_time_steps.append(termination_time_steps)
                all_episodic_returns.append(episodic_returns)
    
    steps, avg_ret, stderr_ret = avg_return_curve(all_termination_time_steps, all_episodic_returns, int_space, total_steps)
    plt.fill_between(steps, avg_ret - stderr_ret, avg_ret + stderr_ret, color="tab:blue", alpha=0.4)
    plt.plot(steps, avg_ret, linewidth=2.0)

    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel(f"Average Episodic Return", fontsize=14)
    plt.title(r"Stream-AC" + f" in {env_name}")
    plt.savefig(f"{env_name}.pdf")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--int_space', type=int, default=50_000)
    parser.add_argument('--total_steps', type=int, default=2_000_000)
    args = parser.parse_args()
    main(args.data_dir, args.int_space, args.total_steps)