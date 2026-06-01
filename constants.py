"""Shared constants. Kept dependency-free (only stdlib + pytz) so any module
can import it without risk of an import cycle."""
import pytz

# Single source of truth for the market timezone used across the project.
IST = pytz.timezone("Asia/Kolkata")
