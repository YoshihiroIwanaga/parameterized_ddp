import math
import typing

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import ArtistAnimation
from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D

wheel_base = 2.7
vehicle_width = 1.70
wheel_length = 0.9
wheel_width = 0.3
wheel_r = 0.57
b1 = 2.43
L2 = 6.2
x = 2.0

body_center_to_wind_shield = Affine2D().translate(0, 0.5 + wheel_base)
body_center_to_rear_window = Affine2D().translate(0, 0.0)
body_center_to_right_headlight = Affine2D().translate(-0.7, x + wheel_base)
body_center_to_left_headlight = Affine2D().translate(0.7, x + wheel_base)


def relative_rectangle(w: float, h: float, center_tf, **kwargs) -> Rectangle:
    rect_origin_to_center = Affine2D().translate(w / 2, h)
    return Rectangle(
        (0, 0), w, h, transform=rect_origin_to_center.inverted() + center_tf, **kwargs
    )


def relative_wheel(w: float, h: float, center_tf, **kwargs) -> Rectangle:
    rect_origin_to_center = Affine2D().translate(w / 2, h / 2)
    return Rectangle(
        (0, 0), w, h, transform=rect_origin_to_center.inverted() + center_tf, **kwargs
    )


def relative_polygon(points, center_tf, **kwargs) -> Polygon:
    return Polygon(xy=points, transform=center_tf, closed=True, **kwargs)


def relative_board(w: float, h: float, center_tf, **kwargs) -> Rectangle:
    rect_origin_to_center = Affine2D().translate(w / 2, h / 8)
    return Rectangle(
        (0, 0), w, h, transform=rect_origin_to_center.inverted() + center_tf, **kwargs
    )


def relative_fork(w: float, h: float, center_tf, **kwargs) -> Rectangle:
    rect_origin_to_center = Affine2D().translate(w / 2, h / 2)
    return Rectangle(
        (0, 0), w, h, transform=rect_origin_to_center.inverted() + center_tf, **kwargs
    )


def draw_car(
    posi_x,
    posi_y,
    yaw,
    steering_angle,
    trans_,
    fill_=True,
    lw_=1.25,
    ls_="solid",
    zorder_=1,
) -> typing.List[Rectangle]:  # -> list[Any]:
    to_body_center_tf = (
        Affine2D().rotate(yaw - math.pi / 2).translate(posi_x, posi_y) + trans_
    )
    body_center_to_left_front_wheel = (
        Affine2D().rotate(steering_angle).translate(-vehicle_width / 2, wheel_base)
    )
    body_center_to_right_front_wheel = (
        Affine2D().rotate(steering_angle).translate(vehicle_width / 2, wheel_base)
    )
    body_center_to_left_rear_wheel = (
        Affine2D().rotate(0).translate(-vehicle_width / 2, 0.0)
    )
    body_center_to_right_rear_wheel = (
        Affine2D().rotate(0).translate(vehicle_width / 2, 0.0)
    )
    body_center_to_outer = Affine2D().translate(0.0, x + wheel_base)
    wind_shield = relative_rectangle(
        vehicle_width,
        1.0,
        body_center_to_wind_shield + to_body_center_tf,
        facecolor="skyblue",
        edgecolor="black",
        fill=fill_,
        lw=lw_,
        ls=ls_,
        zorder=zorder_,
        alpha=0.8,
    )
    rear_window = relative_rectangle(
        vehicle_width,
        0.5,
        body_center_to_rear_window + to_body_center_tf,
        facecolor="skyblue",
        edgecolor="black",
        fill=fill_,
        lw=lw_,
        ls=ls_,
        zorder=zorder_,
        alpha=0.8,
    )
    right_head_light = relative_rectangle(
        0.6,
        0.3,
        body_center_to_right_headlight + to_body_center_tf,
        facecolor="yellow",
        edgecolor="black",
        fill=fill_,
        lw=lw_,
        ls=ls_,
        zorder=zorder_,
        alpha=0.8,
    )
    left_head_light = relative_rectangle(
        0.6,
        0.3,
        body_center_to_left_headlight + to_body_center_tf,
        facecolor="yellow",
        edgecolor="black",
        fill=fill_,
        lw=lw_,
        ls=ls_,
        zorder=zorder_,
        alpha=0.8,
    )
    outer = relative_rectangle(
        b1,
        L2,
        body_center_to_outer + to_body_center_tf,
        facecolor="gray",
        edgecolor="black",
        fill=fill_,
        lw=lw_,
        ls=ls_,
        zorder=zorder_,
        alpha=0.8,
    )
    left_front_wheel = relative_wheel(
        wheel_width,
        wheel_length,
        body_center_to_left_front_wheel + to_body_center_tf,
        facecolor="black",
        edgecolor="black",
        fill=fill_,
        lw=lw_,
        ls=ls_,
    )
    right_front_wheel = relative_wheel(
        wheel_width,
        wheel_length,
        body_center_to_right_front_wheel + to_body_center_tf,
        facecolor="black",
        edgecolor="black",
        fill=fill_,
        lw=lw_,
        ls=ls_,
        zorder=zorder_,
    )
    left_rear_wheel = relative_wheel(
        wheel_width,
        wheel_length,
        body_center_to_left_rear_wheel + to_body_center_tf,
        facecolor="black",
        edgecolor="black",
        fill=fill_,
        lw=lw_,
        ls=ls_,
        zorder=zorder_,
    )
    right_rear_wheel = relative_wheel(
        wheel_width,
        wheel_length,
        body_center_to_right_rear_wheel + to_body_center_tf,
        facecolor="black",
        edgecolor="black",
        fill=fill_,
        lw=lw_,
        ls=ls_,
        zorder=zorder_,
    )
    patches_list = [
        left_front_wheel,
        right_front_wheel,
        left_rear_wheel,
        right_rear_wheel,
        outer,
        wind_shield,
        rear_window,
        right_head_light,
        left_head_light,
    ]

    return patches_list


def create_gif(
    filename: str,
    xs,
    us,
    interpolated_xs,
    interpolated_us,
    xs_opt: np.ndarray,
    horizon: int,
    x_ref: np.ndarray,
    lw: float,
):

    fig = plt.figure(figsize=(15, 9), dpi=120)
    ax0 = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(3, 6, 4)
    ax1.plot((horizon + 1) * [x_ref[0]], "green", linewidth=lw)
    ax2 = fig.add_subplot(3, 6, 5)
    ax2.plot((horizon + 1) * [x_ref[1]], "green", linewidth=lw)
    ax3 = fig.add_subplot(3, 6, 6)
    ax3.plot((horizon + 1) * [x_ref[2]], "green", linewidth=lw)
    ax4 = fig.add_subplot(3, 6, 10)
    ax4.plot((horizon + 1) * [x_ref[3]], "green", linewidth=lw)
    ax5 = fig.add_subplot(3, 6, 11)
    ax5.plot((horizon + 1) * [x_ref[4]], "green", linewidth=lw)
    ax6 = fig.add_subplot(3, 6, 16)
    ax7 = fig.add_subplot(3, 6, 17)
    image_list = []

    initial_state_patches = draw_car(0, 0, 0.0, 0.0, ax0.transData)
    for patch_ in initial_state_patches:
        ax0.add_patch(patch_)

    target_state_patches = draw_car(
        x_ref[0], x_ref[1], x_ref[2], x_ref[4], ax0.transData, ls_="dotted"
    )
    for patch_ in target_state_patches:
        ax0.add_patch(patch_)

    ax0.set_aspect(1)
    ax0.set_xlim(
        min(xs_opt[0, :]) - 5.0,
        max(xs_opt[0, :]) + 5.0,
    )
    ax0.set_ylim(
        min(xs_opt[1, :]) - 5.0,
        max(xs_opt[1, :]) + 5.0,
    )

    for x, u, X, U in zip(xs, us, interpolated_xs, interpolated_us):
        image0_0 = ax0.plot(
            X[0, :],
            X[1, :],
            color="blue",
            linewidth=lw,
        )
        image0_1 = ax0.plot(
            x[0, :],
            x[1, :],
            color="red",
            linestyle="dashed",
            linewidth=lw,
        )
        image1_0 = ax1.plot(
            X[0, :],
            color="blue",
            linewidth=lw,
        )
        image1_1 = ax1.plot(
            x[0, :],
            color="red",
            linestyle="dashed",
            linewidth=lw,
        )
        image2_0 = ax2.plot(
            X[1, :],
            color="blue",
            linewidth=lw,
        )
        image2_1 = ax2.plot(
            x[1, :],
            color="red",
            linestyle="dashed",
            linewidth=lw,
        )
        image3_0 = ax3.plot(
            X[2, :],
            color="blue",
            linewidth=lw,
        )
        image3_1 = ax3.plot(
            x[2, :],
            color="red",
            linestyle="dashed",
            linewidth=lw,
        )
        image4_0 = ax4.plot(
            X[3, :],
            color="blue",
            linewidth=lw,
        )
        image4_1 = ax4.plot(
            x[3, :],
            color="red",
            linestyle="dashed",
            linewidth=lw,
        )
        image5_0 = ax5.plot(
            X[4, :],
            color="blue",
            linewidth=lw,
        )
        image5_1 = ax5.plot(
            x[4, :],
            color="red",
            linestyle="dashed",
            linewidth=lw,
        )
        image6_0 = ax6.plot(
            U[0, :],
            color="blue",
            linewidth=lw,
        )
        image6_1 = ax6.plot(
            u[0, :],
            color="red",
            linestyle="dashed",
            linewidth=lw,
        )
        image7_0 = ax7.plot(
            U[1, :],
            color="blue",
            linewidth=lw,
        )
        image7_1 = ax7.plot(
            u[1, :],
            color="red",
            linestyle="dashed",
            linewidth=lw,
        )

        image_list.append(
            image0_0
            + image0_1
            + image1_0
            + image1_1
            + image2_0
            + image2_1
            + image3_0
            + image3_1
            + image4_0
            + image4_1
            + image5_0
            + image5_1
            + image6_0
            + image6_1
            + image7_0
            + image7_1
        )
    ani = ArtistAnimation(fig, image_list, interval=300)
    ani.save(filename, writer="pillow")
