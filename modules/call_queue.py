"""The generation lock."""

from modules import fifo_lock

queue_lock = fifo_lock.FIFOLock()
"""Serializes GPU work: the jobs worker, ``/sdapi`` generation and GPU actions."""
