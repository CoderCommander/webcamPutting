"""Tests for GPU probe and initialization."""

from birdman_putting.gpu import get_device_count, is_cuda_available


class TestGpuProbe:
    def test_is_cuda_available_returns_bool(self) -> None:
        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_get_device_count_returns_int(self) -> None:
        result = get_device_count()
        assert isinstance(result, int)
        assert result >= 0

    def test_init_cuda_graceful_failure(self) -> None:
        from birdman_putting.gpu import init_cuda

        # Should not raise even if no CUDA or no matching device
        result = init_cuda("NonExistentGPU_XYZ_999")
        assert isinstance(result, bool)
