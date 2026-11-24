#!/bin/bash
#
# Copyright 2026 Junyi Peng (pengjy@fit.vut.cz)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# local/simulate_all.sh <rtcsdd_dir> <data_dir> <sim_dir> [nj]
#
# Stage 2: render every line "<subset> <preset> <seed>" of conf/sim_variants.txt:
# the offline recordings of <subset> through local/simulate_rtc.py, written to
# <sim_dir>/<subset>_<preset>/ under their relative paths, on nj CPU processes.
# Finished files are skipped, so an interrupted run can be restarted.

set -e

rtcsdd_dir=$1
data=$2
sim_dir=$3
nj=${4:-16}

while read -r subset preset seed; do
  echo "== ${subset} ${preset} (seed ${seed})"
  noise=musan_noise
  [ "${preset}" = reverb ] && noise=musan_reverb
  python local/simulate_rtc.py --preset ${preset} --seed ${seed} \
    --nproc ${nj} --skip_existing \
    --list ${data}/${subset}_offline/wav.scp \
    --in_root ${rtcsdd_dir}/wav/${subset} \
    --out_root ${sim_dir}/${subset}_${preset} \
    --noise_scp ${data}/noise/${noise}.scp \
    --rir_scp ${data}/noise/rirs.scp \
    --event_dir ${data}/noise < /dev/null
done < <(grep -v '^#' conf/sim_variants.txt)
