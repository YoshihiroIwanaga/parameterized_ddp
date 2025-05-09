import matplotlib.pyplot as plt


def plot_xs(
    xs_org,
    xs_interpolated,
    x_ref,
    steps_per_knot,
    time_step_on_knot,
    horizon,
    interpolation_color="blue",
) -> None:
    n = xs_org.shape[0]
    knot_frames = [0]
    f = 0
    for steps in steps_per_knot:
        f += steps
        knot_frames.append(f)

    fig = plt.figure(figsize=(15, 3.5 * n))
    for i in range(n):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.plot(xs_interpolated[i, :], color=interpolation_color)
        ax.plot(
            knot_frames,
            xs_interpolated[i, time_step_on_knot],
            "o",
            color=interpolation_color,
        )
        ax.plot(xs_org[i, :], "red", linestyle="dashed")
        ax.plot((horizon + 1) * [x_ref[i]], "green")
    plt.show()


def plot_us(
    us_org,
    us_interpolated,
    steps_per_knot,
    time_step_on_knot,
    horizon,
    interpolation_color="blue",
) -> None:
    m = us_org.shape[0]
    knot_frames = [0]
    f = 0
    for steps in steps_per_knot:
        f += steps
        knot_frames.append(f)
    fig = plt.figure(figsize=(15, 3.5 * m))
    for i in range(m):
        ax = fig.add_subplot(m, 1, i + 1)
        ax.plot(us_interpolated[i, :], color=interpolation_color)
        ax.plot(
            knot_frames[:-1],
            us_interpolated[i, time_step_on_knot[:-1]],
            "o",
            color=interpolation_color,
        )
        ax.plot(us_org[i, :], "red", linestyle="dashed")
    plt.show()
