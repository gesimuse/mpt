FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg imagemagick fonts-dejavu-core cron tini \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i 's/rights="none" pattern="@\*"/rights="read,write" pattern="@*"/' \
       /etc/ImageMagick-6/policy.xml || true

WORKDIR /app

# MoneyPrinterTurbo
RUN git clone --depth 1 https://github.com/harry0703/MoneyPrinterTurbo.git
RUN pip install --no-cache-dir -r MoneyPrinterTurbo/requirements.txt

# Autopilot deps
RUN pip install --no-cache-dir requests google-api-python-client google-auth google-auth-oauthlib

COPY autopilot.py niches.json ./
COPY run_schedule.sh /run_schedule.sh
RUN chmod +x /run_schedule.sh

ENV MPT_DIR=/app/MoneyPrinterTurbo
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/run_schedule.sh"]
