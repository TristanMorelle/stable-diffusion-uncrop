ARG IMAGE=hlky/sd-webui:base

FROM ${IMAGE}

WORKDIR /workdir

SHELL ["/bin/bash", "-c"]

ENV PYTHONPATH=/sd

EXPOSE 8501
COPY ./data/DejaVuSans.ttf /usr/share/fonts/truetype/
COPY ./data/ /sd/data/
copy ./images/ /sd/images/
copy ./scripts/ /sd/scripts/
copy ./ldm/ /sd/ldm/
copy ./frontend/ /sd/frontend/
copy ./configs/ /sd/configs/
copy ./.streamlit/ /sd/.streamlit/
COPY ./entrypoint.sh /sd/
ENTRYPOINT /sd/entrypoint.sh

RUN mkdir -p ~/.streamlit/
RUN echo "[general]"  > ~/.streamlit/credentials.toml
RUN echo "email = \"\""  >> ~/.streamlit/credentials.toml
