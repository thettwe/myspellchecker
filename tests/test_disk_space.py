from collections import namedtuple
from unittest.mock import patch

import pytest

from myspellchecker.core.exceptions import InsufficientStorageError
from myspellchecker.data_pipeline.config import PipelineConfig
from myspellchecker.data_pipeline.pipeline import Pipeline

# Match shutil.disk_usage return type (named tuple with .total/.used/.free)
_DiskUsage = namedtuple("usage", ["total", "used", "free"])


class TestDiskSpace:
    def test_pipeline_insufficient_space_work_dir(self, tmp_path, mock_console):
        """Test failure when work_dir has insufficient space."""
        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=tmp_path)

        # Mock disk_usage to return 10MB free space (below default requirement)
        with patch("shutil.disk_usage") as mock_disk_usage:
            mock_disk_usage.return_value = _DiskUsage(2000, 1000, 10 * 1024 * 1024)

            with pytest.raises(InsufficientStorageError, match="Insufficient disk space"):
                pipeline.build_database(
                    input_files=[], database_path=tmp_path / "out.db", sample=True
                )

    def test_pipeline_insufficient_space_output_dir(self, tmp_path, mock_console):
        """Test failure when output DB directory has insufficient space."""
        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=tmp_path)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock disk_usage logic
        # First call (work_dir) -> OK (2GB)
        # Second call (output_db) -> Fail (100MB)

        with patch("shutil.disk_usage") as mock_disk_usage:
            mock_disk_usage.side_effect = [
                _DiskUsage(5000, 1000, 2 * 1024 * 1024 * 1024),  # 2GB free for work_dir
                _DiskUsage(5000, 1000, 100 * 1024 * 1024),  # 100MB free for output_db
            ]

            with pytest.raises(InsufficientStorageError, match="Insufficient disk space"):
                pipeline.build_database(
                    input_files=[], database_path=output_dir / "out.db", sample=True
                )

    def test_pipeline_sufficient_space(self, tmp_path, mock_console):
        """Test pass when space is sufficient."""
        config = PipelineConfig(console=mock_console)
        pipeline = Pipeline(config=config, work_dir=tmp_path)

        with patch("shutil.disk_usage") as mock_disk_usage:
            # 100GB free (must exceed disk_space_check_mb default of 51200 MB)
            mock_disk_usage.return_value = _DiskUsage(
                200 * 1024 * 1024 * 1024,
                100 * 1024 * 1024 * 1024,
                100 * 1024 * 1024 * 1024,
            )

            # Should run without storage error (will likely finish sample build)
            pipeline.build_database(input_files=[], database_path=tmp_path / "out.db", sample=True)
            assert (tmp_path / "out.db").exists()
