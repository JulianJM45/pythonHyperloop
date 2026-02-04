# %%
import sys

sys.path.insert(0, "..")
from modules.components import fit_B1, get_b0, load_data, plot_data, test_B1

# %%
file_names = [
    "Line4yamamura_long_y10.csv",
    "Line4yamamura_y05.csv",
    "Line4yamamura_y06.csv",
    "Line4yamamura_y07.csv",
    "Line4yamamura_y06.csv",
    "Line4yamamura_y09.csv",
    "Line4yamamura_y10.csv",
    "Line4yamamura_y15.csv",
    "Line4yamamura_y20.csv",
    "Line4yamamura_y30.csv",
]
file_name = file_names[0]
# static_file_name = file_name.replace('.csv', '_static.csv')
data, time_steps = load_data(file_name)
# data, time_steps = merge_csv_files(static_file_name, file_name)
# plot_data(data, time_steps[2])


def main():
    process_data(data, time_steps)
    # for file_name in file_names:
    #     print(f"Processing {file_name}")
    #     # static_file_name = file_name.replace('.csv', '_static.csv')
    #     # static_dynamics_comparison(static_file_name, file_name)
    #     data, time_steps = load_data(file_name)
    #     process_data(data, time_steps)


# %%
def process_data(data, time_steps):
    plot_data(data, time_steps[-10])

    x_interval = [30, 70]
    B0 = get_b0(data, time_steps[0], x_interval)
    print(f"Mean B value between x={x_interval[0]} and x={x_interval[1]}: {B0:.4f} T")
    test_B1(data, time_steps[-10], x0=90, x1=68, amp=0.036, a=0.1)
    # fit_B1(data, time_steps[-1], B0, fitrange=[68, 140], x0=40, amp=0.036, a=0.2)


def static_dynamics_comparison(static_file_name, file_name):
    static_data, static_time_steps = load_data(static_file_name)
    data, time_steps = load_data(file_name)
    x_interval = [30, 70]
    static_B0 = get_b0(static_data, static_time_steps[0], x_interval)
    # print(f"Mean static B value between x={x_interval[0]} and x={x_interval[1]}: {static_B0:.4f} T")
    B0 = get_b0(data, time_steps[0], x_interval)
    # print(f"Mean B value between x={x_interval[0]} and x={x_interval[1]}: {B0:.4f} T")
    print(
        f"procentual difference for file {file_name}: {(B0 - static_B0) / static_B0 * 100:.2f}%"
    )


if __name__ == "__main__":
    main()
