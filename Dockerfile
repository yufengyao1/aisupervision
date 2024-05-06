# FROM lingoace_cuda11.4:latest
FROM class_base:latest
# FROM lg_class_cpu:latest
# FROM base_cpu:latest

WORKDIR /data

COPY . /data/

# RUN set -ex \
#     && source ~/.bashrc \
#     && pip install -r requirements.txt \
#     && ln -s /root/miniconda3/bin/ffmpeg /usr/local/bin/ \
#     && ln -s /root/miniconda3/bin/ffprobe /usr/local/bin/

CMD ["python","-u","main.py"]