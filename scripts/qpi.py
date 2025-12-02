from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

# import muscopy.cfg as mcfg
# from muscopy.cfg import OffsetRegions

# if mcfg._cp:
#     import cupy as xp
# else:
#     import numpy as xp

import numpy as xp


@dataclass
class QPIParameters:
    wavelength: float
    NA: float
    img_shape: tuple[int, int]
    pixelsize: float
    offaxis_center: tuple[int, int]
    n_sol: float = 1.33

    @cached_property
    def img_center(self) -> tuple[int, int]:
        return (self.img_shape[0] // 2, self.img_shape[1] // 2)

    @cached_property
    def dim(self) -> int:
        return self.img_shape[0]

    @cached_property
    def freq_per_pixel(self) -> float:
        return 1 / (self.pixelsize * self.dim)

    @cached_property
    def k_per_pixel(self) -> float:
        return 2 * np.pi * self.freq_per_pixel

    @cached_property
    def aperturesize(self) -> int:
        return 2 * round(self.NA / self.wavelength / self.freq_per_pixel) + 1

    @cached_property
    def fi_mag(self) -> float:
        """|f| of the light

        Returns:
            float: the magnitude of the light vector
        """
        return self.n_sol / self.wavelength / self.freq_per_pixel

    @cached_property
    def imgpx_unit(self) -> float:
        """The image pixel unit in the synthetic aperture plane

        Returns:
            float: the image pixel unit in the synthetic aperture plane
        """
        return self.pixelsize * self.img_shape[0] / (2 * self.aperturesize + 1 - mcfg.EDGE_SIZE)

    @cached_property
    def Hologram2F(self) -> float:
        """Fourier factor from hologram to spectrum

        Returns:
            float: factor from hologram to spectrum
        """
        return (self.pixelsize / self.k_per_pixel) ** 0.5

    @cached_property
    def S2F(self) -> float:
        """Fourier factor from spectrum to complex field

        Returns:
            float: factor from spectrum to complex field
        """
        return (self.imgpx_unit / self.k_per_pixel) ** 0.5

    @cached_property
    def F2S(self) -> float:
        """Fourier factor from complex field to spectrum

        Returns:
            float: factor from complex field to spectrum
        """
        return (self.k_per_pixel / self.imgpx_unit) ** 0.5

    def _print_all_parameters(self):
        """print all parameters of the Parameters class"""
        for key, value in vars(self).items():
            print(f"{key}={value}")

    def print_all_parameters(self):
        """print all parameters of the Parameters class"""
        self._print_all_parameters()
        # calculated parameters
        print(f"img_center={self.img_center}")
        print(f"dim={self.dim}")
        print(f"freq_per_pixel={self.freq_per_pixel}")
        print(f"k_per_pixel={self.k_per_pixel}")
        print(f"aperturesize={self.aperturesize}")
        print(f"fi_mag={self.fi_mag}")
        print(f"imgpx_unit={self.imgpx_unit}")
        print(f"Hologram2F={self.Hologram2F}")
        print(f"S2F={self.S2F}")
        print(f"F2S={self.F2S}")


def make_disk(center: tuple[int, int], radius: float, array_shape: tuple[int, int], highpass: bool = False) -> xp.array:
    """make disk mask for filtering

    Args:
        center (tuple[int, int]): center position of the disk mask
        radius (float): radius of the disk mask
        array_shape (tuple[int, int]): shape of the array
        highpass (bool, optional): Filter low frequency or not. Defaults to False.

    Returns:
        xp.array: disk mask
    """
    if isinstance(array_shape, int):
        array_shape = (array_shape, array_shape)
    xx, yy = xp.meshgrid(xp.arange(array_shape[0]), xp.arange(
        array_shape[1]), indexing="ij")
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    if highpass:
        disk = circle > radius**2
    else:
        disk = circle < radius**2
    return disk


def crop_array(array: xp.array, center: tuple[int, int], width: int) -> xp.array:
    """internal method. crop the array with specified center and width.

    Args:
        array (xp.array): array to be cropped
        center (tuple of int): center position of the cropped array
        width (int): width of the cropped array

    Returns:
        xp.array: cropped array
    """
    return array[
        center[0] - width // 2: center[0] + width // 2 + 1,
        center[1] - width // 2: center[1] + width // 2 + 1,
    ]


def get_spectrum(array: xp.array, params: QPIParameters, crop_center: bool = False, c_r: int = 5) -> xp.array:
    """internal method. get the spectrum of the hologram array

    Args:
        array (xp.array): input array
        params (QPIParameters): QPIParameters class
        crop_center (bool, optional): crop the center of the array or not. Defaults to False.

    Returns:
        xp.array: cropped spectrum of the hologram array
    """
    array_fft = xp.fft.fftshift(xp.fft.fft2(array))
    mask = make_disk(params.offaxis_center,
                     params.aperturesize / 2, params.img_shape)
    array_fft = array_fft * mask

    if crop_center:
        mask_highpass = make_disk(
            params.offaxis_center, c_r, params.img_shape, highpass=True)
        array_fft = array_fft * mask_highpass

    array_fft = crop_array(
        array_fft, params.offaxis_center, params.aperturesize)

    return array_fft


def get_field(array: xp.array, params: QPIParameters, crop_center: bool = False, c_r: int = 5) -> xp.array:
    """internal method. get the electric field from the hologram array

    Args:
        array (xp.array): input array
        params (QPIParameters): QPIParameters class
        crop_center (bool, optional): crop the center of the array or not. Defaults to False.

    Returns:
        xp.array: field from the hologram array
    """
    array_fft = get_spectrum(array, params, crop_center, c_r)
    array = xp.fft.ifft2(xp.fft.ifftshift(array_fft))
    return array


def correct_offset(array, offset_regs: OffsetRegions, phase: bool = True, amplitude: bool = True) -> xp.array:
    """internal method. correct phase and amplitude offset

    Args:
        array (NDArray): input complex array
        offset_regs (OffsetRegions): regions for offset calculation

    Returns:
        xp.array: corrected array
    """
    if offset_regs is None:
        return array
    phase_offset_list = []
    amplitude_offset_list = []
    for region in offset_regs:
        phase_offset_list.append(xp.mean(
            xp.angle(array[region[0][0]: region[0][1], region[1][0]: region[1][1]])))
        amplitude_offset_list.append(
            xp.mean(xp.abs(array[region[0][0]: region[0][1], region[1][0]: region[1][1]])))
    if phase:
        phase_offset = xp.mean(xp.array(phase_offset_list))  # - xp.pi / 2
    else:
        phase_offset = 0
    if amplitude:
        amplitude_offset = xp.mean(xp.array(amplitude_offset_list))
    else:
        amplitude_offset = 1

    array = array * xp.exp(-1j * phase_offset) / amplitude_offset

    return array


def QPI(
    array: xp.array,
    reference: xp.array,
    params: QPIParameters,
) -> xp.array:
    """Quantitative phase imaging (QPI) calculation

    Args:
        array (xp.array): on-axis hologram
        reference (xp.array): off-axis hologram
        params (QPIParameters): QPIParameters class

    Returns:
        xp.array: QPI phase image
    """
    assert array.shape == reference.shape

    array_field = get_field(array, params)
    ref_array_field = get_field(reference, params)

    array_div = array_field / ref_array_field

    # remove phase and amplitude offset
    if mcfg.OFFSET_REGS is not None:
        array_div = correct_offset(array_div, mcfg.OFFSET_REGS)

    dif_phase = xp.angle(array_div)

    return dif_phase


def MIPQPI(
    array_on: xp.array,
    array_off: xp.array,
    params: QPIParameters,
    crop_center: bool = False,
    **kwargs,
) -> xp.array:
    """Mid-infrared photothermal quantitative phase imaging (MIP-QPI) calculation

    Args:
        array_on (xp.array): on-axis hologram
        array_off (xp.array): off-axis hologram
        params (QPIParameters): QPIParameters class
        crop_center (bool, optional): crop the center of the array or not. Defaults to False.

    Returns:
        xp.array: MIP-QPI phase image
    """
    assert array_on.shape == array_off.shape
    array_on_field = get_field(
        array_on, params, crop_center=crop_center, **kwargs)
    array_off_field = get_field(
        array_off, params, crop_center=crop_center, **kwargs)

    array_div = array_on_field / array_off_field

    if mcfg.MIP_CENTER is not None:
        center_phase = xp.mean(
            xp.angle(
                array_div[mcfg.MIP_CENTER[0][0]: mcfg.MIP_CENTER[0]
                          [1], mcfg.MIP_CENTER[1][0]: mcfg.MIP_CENTER[1][1]]
            )
        )
        if center_phase < 0:
            array_div = 1 / array_div

    # remove phase and amplitude offset
    if mcfg.OFFSET_REGS is not None:
        array_div = correct_offset(array_div, mcfg.OFFSET_REGS, phase=True)

    dif_phase = xp.angle(array_div)

    return dif_phase


def _get_dc_ac(
    hologram: NDArray,
    params: QPIParameters,
) -> tuple[NDArray, NDArray]:
    """Calculates the DC intensity of the object.

    Args:
        hologram (NDArray): The hologram of the object.
        params (mus.QPIParameters): The parameters of the QPI.

    Returns:
        NDArray: The DC and AC component of the object.
    """
    scale_factor = params.aperturesize / params.img_shape[0]
    fft = xp.fft.fftshift(xp.fft.fft2(hologram))
    dc_disk = make_disk(params.img_center,
                        params.aperturesize // 2, params.img_shape)
    ac_disk = make_disk(params.offaxis_center,
                        params.aperturesize // 2, params.img_shape)

    dc_fft = fft * dc_disk
    ac_fft = fft * ac_disk

    dc_cropped = crop_array(dc_fft, params.img_center, params.aperturesize)
    ac_cropped = crop_array(ac_fft, params.offaxis_center, params.aperturesize)

    dc = xp.fft.ifft2(xp.fft.ifftshift(dc_cropped)) * scale_factor**2
    ac = xp.fft.ifft2(xp.fft.ifftshift(ac_cropped)) * scale_factor**2

    return dc, ac


def _get_visibility(
    dc: NDArray,
    ac: NDArray,
) -> NDArray:
    """Calculates the visibility of the object.

    Args:
        dc (NDArray): dc component of the hologram
        ac (NDArray): ac component of the hologram

    Returns:
        NDArray: pixel-wise visibility of the hologram
    """
    visibility = xp.abs(ac) / xp.abs(dc) * 2
    return visibility


def _get_phase_noise(
    visibility: NDArray,
    aperturesize: int,
    dc_intensity: NDArray,
    sensorsize: tuple[int, int],
) -> NDArray:
    """Calculates the phase noise in a given array.

    Args:
        visibility (NDArray): The visibility of the object.
        aperturesize (int): The size of the aperture.
        dc_intensity (NDArray): The DC intensity of the object.
        sensorsize (tuple[int, int]): The size of the sensor.

    Returns:
        NDArray: The phase noise
    """
    if not visibility.shape == dc_intensity.shape:
        raise ValueError(
            "Visibility and DC intensity must have the same shape")
    aperture_area = xp.pi * (aperturesize / 2) ** 2
    sensor_area = sensorsize[0] * sensorsize[1]
    phase_noise = xp.sqrt(2 * aperture_area /
                          (visibility**2 * dc_intensity * sensor_area))

    return phase_noise


def calc_visibility(
    hologram: NDArray,
    params: QPIParameters,
) -> NDArray:
    """Calculates the visibility of the object.

    Args:
        hologram (NDArray): The hologram of the object.
        params (mus.QPIParameters): The parameters of the QPI.

    Returns:
        NDArray: pixel-wise visibility of the hologram
    """
    dc, ac = _get_dc_ac(hologram, params)
    visibility = _get_visibility(dc, ac)

    return visibility


def calc_phase_noise(
    hologram: NDArray,
    params: QPIParameters,
) -> NDArray:
    """Calculates the phase noise in a given array.

    Args:
        hologram (NDArray): The hologram of the object.
        params (mus.QPIParameters): The parameters of the QPI.

    Returns:
        NDArray: The phase noise
    """
    dc, ac = _get_dc_ac(hologram, params)
    visibility = _get_visibility(dc, ac)
    phase_noise = _get_phase_noise(
        visibility, params.aperturesize, xp.abs(dc), params.img_shape)

    return phase_noise
