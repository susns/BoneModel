#!/bin/bash

opengr_url='https://github.com/STORM-IRIT/OpenGR.git'
opengr_dir='./OpenGR'
if [[ ! -a ${opengr_dir} ]]; then 
  git clone ${opengr_url}
fi

if [[ -d ${opengr_dir} ]] && cd ${opengr_dir}; then
  mkdir build && cd build
  cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=./install
  make install
  cd install/scripts
  ./run-example.sh
  echo "Congratulations for install finished!"
  export PATH=$PATH:$(pwd)/bin
else
  echo "can't find OpenGR dir, something error."
fi



