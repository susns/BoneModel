#!/bin/bash

opengr_url='https://github.com/STORM-IRIT/OpenGR.git'
opengr_dir='./OpenGR'
if [[ ! -a ${opengr_dir} ]]; then 
  git clone ${opengr_url}
fi

if [[ -d ${opengr_dir} ]] && cd ${opengr_dir}; then
  mkdir build && cd build || exit
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
  make install
  cd install/scripts || exit
  ./run-example.sh
  echo "Congratulations for install finished!"
  export PATH=$(pwd)/bin:${PATH}
else
  echo "can't find OpenGR dir, something error."
fi



