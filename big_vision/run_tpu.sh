# Copyright 2024 Big Vision Authors.
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

#!/bin/bash

if [ ! -d "bv_venv" ]
then
  sudo apt-get update
  sudo apt install -y python3-venv
  python3 -m venv bv_venv
  . bv_venv/bin/activate

  pip install -U pip  # Yes, really needed.
  # NOTE: doesn't work when in requirements.txt -> cyclic dep
  pip install "jax[tpu]>=0.4.25" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install -r big_vision/requirements.txt
else
  . bv_venv/bin/activate
fi

if [ $# -ne 0 ]
then
  env TFDS_DATA_DIR=$TFDS_DATA_DIR BV_JAX_INIT=1 python3 -m "$@"
fi
