# Copyright 2017 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from ament_flake8.main import main_with_errors
import pytest

# The project config is ament_flake8's default configuration with the quote
# style adapted to this codebase (see the comment in .flake8).
CONFIG_FILE = Path(__file__).resolve().parents[1] / ".flake8"


@pytest.mark.flake8
@pytest.mark.linter
def test_flake8():
    rc, errors = main_with_errors(
        argv=["--config", str(CONFIG_FILE), str(CONFIG_FILE.parent)]
    )
    assert rc == 0, (
        "Found %d code style errors / warnings:\n" % len(errors)
        + "\n".join(errors)
    )
