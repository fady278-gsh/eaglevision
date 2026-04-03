"""
kafka_pub.py  –  Kafka Producer for CV Microservice
=====================================================
Publishes equipment event JSON payloads to a Kafka topic.
Uses equipment_id as the partition key → all events for one machine
land on the same partition → ordered consumer processing.
"""

from __future__ import annotations
import json
import logging
from typing import Optional

log = logging.getLogger("kafka_pub")


class KafkaPublisher:
    """
    Thin wrapper around confluent_kafka Producer.

    Retry / error handling
    ----------------------
    on_delivery callback logs delivery errors without blocking the CV pipeline.
    If Kafka is unavailable, messages queue in the local producer buffer
    (queue.buffering.max.messages default = 100,000) giving a soft buffer.
    """

    def __init__(self, bootstrap_servers: str, topic: str):
        self.topic = topic
        try:
            from confluent_kafka import Producer
            self._producer = Producer({
                "bootstrap.servers"            : bootstrap_servers,
                "queue.buffering.max.messages" : 100_000,
                "queue.buffering.max.ms"       : 50,      # low latency
                "compression.type"             : "lz4",
                "acks"                         : "1",     # leader ack only (speed > durability)
            })
            log.info(f"Kafka producer connected → {bootstrap_servers} / {topic}")
        except ImportError:
            raise RuntimeError("confluent_kafka not installed. Run: pip install confluent-kafka")
        except Exception as e:
            log.warning(f"Kafka producer init failed: {e}. Events will not be streamed.")
            self._producer = None

    def publish(self, payload: dict, key: Optional[str] = None) -> None:
        if self._producer is None:
            return
        try:
            key_bytes = (key or payload.get("equipment_id", "default")).encode()
            value_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self._producer.produce(
                topic    = self.topic,
                key      = key_bytes,
                value    = value_bytes,
                callback = self._delivery_callback,
            )
            self._producer.poll(0)   # non-blocking flush of callbacks
        except Exception as e:
            log.error(f"Kafka produce error: {e}")

    def close(self):
        if self._producer:
            self._producer.flush(timeout=5)
            log.info("Kafka producer flushed and closed.")

    @staticmethod
    def _delivery_callback(err, msg):
        if err:
            log.warning(f"Kafka delivery failed: {err}")
