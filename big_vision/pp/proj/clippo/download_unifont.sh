# Copyright 2022 Big Vision Authors.
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
# This is intended to be run from the big_vision repository root:
#
# bash big_vision/pp/proj/clippo/download_unifont.sh
wget https://unifoundry.com/pub/unifont/unifont-9.0.06/font-builds/unifont-9.0.06.hex.gz https://unifoundry.com/pub/unifont/unifont-9.0.06/font-builds/unifont_upper-9.0.06.hex.gz
gunzip unifont-9.0.06.hex.gz unifont_upper-9.0.06.hex.gz
mv unifont-9.0.06.hex unifont_upper-9.0.06.hex big_vision/pp/proj/clippo/