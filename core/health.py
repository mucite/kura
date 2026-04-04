"""
System Health Checks for Kura Medical
======================================
Validate system readiness before operation.
"""
import logging
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("kura.health")


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    status: str  # "OK", "WARNING", "ERROR"
    message: str
    details: Optional[dict] = None


class SystemHealthChecker:
    """Comprehensive system health validation."""

    def __init__(self):
        self.results: List[HealthCheckResult] = []

    def check_all(self) -> Tuple[bool, List[HealthCheckResult]]:
        """
        Run all health checks.

        Returns:
            Tuple of (all_passed, list of results)
        """
        self.results = []

        # Run all checks
        self.check_python_version()
        self.check_system_resources()
        self.check_disk_space()
        self.check_models()
        self.check_gpu()
        self.check_license_system()
        self.check_configuration()
        self.check_pricing_data()

        # Determine if all critical checks passed
        critical_failures = [r for r in self.results if r.status == "ERROR"]
        all_passed = len(critical_failures) == 0

        return all_passed, self.results

    def check_python_version(self):
        """Check Python version compatibility."""
        version = sys.version_info

        if version.major == 3 and version.minor >= 10:
            self.results.append(HealthCheckResult(
                component="Python Version",
                status="OK",
                message=f"Python {version.major}.{version.minor}.{version.micro}",
                details={"version": f"{version.major}.{version.minor}.{version.micro}"}
            ))
        else:
            self.results.append(HealthCheckResult(
                component="Python Version",
                status="ERROR",
                message=f"Python {version.major}.{version.minor} not supported. Requires Python 3.10+",
                details={"version": f"{version.major}.{version.minor}"}
            ))

    def check_system_resources(self):
        """Check system memory and CPU."""
        try:
            import psutil

            # Memory check
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024 ** 3)
            available_gb = mem.available / (1024 ** 3)

            if available_gb < 2.0:
                status = "ERROR"
                message = f"Low memory: {available_gb:.1f}GB available (need 2GB+)"
            elif available_gb < 4.0:
                status = "WARNING"
                message = f"Marginal memory: {available_gb:.1f}GB available (recommend 4GB+)"
            else:
                status = "OK"
                message = f"Memory OK: {available_gb:.1f}GB / {total_gb:.1f}GB available"

            self.results.append(HealthCheckResult(
                component="Memory",
                status=status,
                message=message,
                details={
                    "total_gb": round(total_gb, 1),
                    "available_gb": round(available_gb, 1),
                    "percent_used": mem.percent
                }
            ))

            # CPU check
            cpu_count = psutil.cpu_count()
            self.results.append(HealthCheckResult(
                component="CPU",
                status="OK",
                message=f"{cpu_count} cores available",
                details={"core_count": cpu_count}
            ))

        except ImportError:
            self.results.append(HealthCheckResult(
                component="System Resources",
                status="WARNING",
                message="psutil not installed - cannot check memory"
            ))

    def check_disk_space(self):
        """Check available disk space."""
        try:
            import psutil

            # Check user data directory
            if platform.system() == "Darwin":
                user_dir = Path.home() / "Documents" / "Kura"
            elif platform.system() == "Windows":
                user_dir = Path.home() / "Documents" / "Kura"
            else:
                user_dir = Path.home() / ".kura"

            usage = psutil.disk_usage(str(user_dir.parent))
            free_gb = usage.free / (1024 ** 3)

            if free_gb < 1.0:
                status = "ERROR"
                message = f"Low disk space: {free_gb:.1f}GB free (need 1GB+)"
            elif free_gb < 5.0:
                status = "WARNING"
                message = f"Low disk space: {free_gb:.1f}GB free (recommend 5GB+)"
            else:
                status = "OK"
                message = f"Disk space OK: {free_gb:.1f}GB free"

            self.results.append(HealthCheckResult(
                component="Disk Space",
                status=status,
                message=message,
                details={
                    "free_gb": round(free_gb, 1),
                    "total_gb": round(usage.total / (1024 ** 3), 1),
                    "percent_used": usage.percent
                }
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component="Disk Space",
                status="WARNING",
                message=f"Could not check disk space: {e}"
            ))

    def check_models(self):
        """Check if AI models are present."""
        if getattr(sys, 'frozen', False):
            base = Path(getattr(sys, '_MEIPASS', os.path.dirname(sys.executable)))
        else:
            base = Path(__file__).parent.parent

        models_dir = base / "models"

        if not models_dir.exists():
            self.results.append(HealthCheckResult(
                component="AI Models",
                status="ERROR",
                message=f"Models directory not found: {models_dir}",
                details={"path": str(models_dir)}
            ))
            return

        # Check for required model files
        required_models = {
            "Whisper": "whisper-large-v3-turbo",
            "LLM": "Llama-3.2-3B-Instruct-4bit"
        }

        missing = []
        for name, dir_name in required_models.items():
            model_path = models_dir / dir_name
            if not model_path.exists():
                missing.append(name)

        if missing:
            self.results.append(HealthCheckResult(
                component="AI Models",
                status="ERROR",
                message=f"Missing models: {', '.join(missing)}",
                details={"missing": missing}
            ))
        else:
            self.results.append(HealthCheckResult(
                component="AI Models",
                status="OK",
                message="All required models present",
                details={"models": list(required_models.keys())}
            ))

    def check_gpu(self):
        """Check GPU/Metal availability."""
        system = platform.system()

        if system == "Darwin":
            # macOS - check for Metal
            try:
                import mlx.core as mx
                # Try to allocate a small array
                test = mx.array([1, 2, 3])
                mx.eval(test)

                self.results.append(HealthCheckResult(
                    component="GPU (Metal)",
                    status="OK",
                    message="Metal GPU available",
                    details={"backend": "Metal", "platform": "macOS"}
                ))
            except Exception as e:
                self.results.append(HealthCheckResult(
                    component="GPU (Metal)",
                    status="ERROR",
                    message=f"Metal GPU not available: {e}",
                    details={"error": str(e)}
                ))

        elif system == "Windows":
            # Windows - check for CUDA or fallback to CPU
            try:
                import torch
                cuda_available = torch.cuda.is_available()

                if cuda_available:
                    device_name = torch.cuda.get_device_name(0)
                    self.results.append(HealthCheckResult(
                        component="GPU (CUDA)",
                        status="OK",
                        message=f"CUDA GPU available: {device_name}",
                        details={"backend": "CUDA", "device": device_name}
                    ))
                else:
                    self.results.append(HealthCheckResult(
                        component="GPU (CUDA)",
                        status="WARNING",
                        message="No CUDA GPU - will use CPU (slower)",
                        details={"backend": "CPU"}
                    ))
            except ImportError:
                self.results.append(HealthCheckResult(
                    component="GPU",
                    status="WARNING",
                    message="PyTorch not loaded yet - GPU check deferred"
                ))

    def check_license_system(self):
        """Check license management system."""
        try:
            from shared.license_manager import LicenseManager

            mgr = LicenseManager()
            status = mgr.verify_locally()

            if status is True:
                msg = "License active (Pro)"
            elif status == "TRIAL":
                count = mgr.get_trial_count()
                remaining = mgr.max_trials - count
                msg = f"Trial mode: {remaining}/{mgr.max_trials} reports remaining"
            else:
                msg = "License expired or invalid"

            if status is True or status == "TRIAL":
                status_code = "OK"
            elif status == "TRIAL_EXPIRED":
                status_code = "WARNING"
            else:
                status_code = "ERROR"

            self.results.append(HealthCheckResult(
                component="License",
                status=status_code,
                message=msg,
                details={"license_status": str(status)}
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component="License",
                status="ERROR",
                message=f"License system error: {e}",
                details={"error": str(e)}
            ))

    def check_configuration(self):
        """Check configuration files."""
        # Check .env file
        user_env = Path.home() / "Documents" / "Kura" / ".env"

        if not user_env.exists():
            self.results.append(HealthCheckResult(
                component="Configuration",
                status="WARNING",
                message=".env file not found - will be created on first run",
                details={"path": str(user_env)}
            ))
        else:
            self.results.append(HealthCheckResult(
                component="Configuration",
                status="OK",
                message="Configuration file present",
                details={"path": str(user_env)}
            ))

    def check_pricing_data(self):
        """Check pricing data availability."""
        try:
            from core.config.loader import load_pricing_data

            current_year = datetime.now().year
            pricing = load_pricing_data(current_year)

            # Check if data is current
            valid_until = pricing.get("_valid_until")
            if valid_until:
                expiry = datetime.strptime(valid_until, "%Y-%m-%d")
                days_until_expiry = (expiry - datetime.now()).days

                if days_until_expiry < 0:
                    status = "WARNING"
                    message = f"Pricing data expired {-days_until_expiry} days ago - update needed"
                elif days_until_expiry < 30:
                    status = "WARNING"
                    message = f"Pricing data expires in {days_until_expiry} days - update soon"
                else:
                    status = "OK"
                    message = f"Pricing data valid (expires in {days_until_expiry} days)"
            else:
                status = "OK"
                message = "Pricing data loaded"

            self.results.append(HealthCheckResult(
                component="Pricing Data",
                status=status,
                message=message,
                details={
                    "year": current_year,
                    "valid_until": valid_until
                }
            ))

        except Exception as e:
            self.results.append(HealthCheckResult(
                component="Pricing Data",
                status="WARNING",
                message=f"Using fallback pricing: {e}",
                details={"error": str(e)}
            ))

    def generate_report(self) -> str:
        """Generate human-readable health report."""
        lines = [
            "=" * 60,
            "Kura Medical System Health Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Platform: {platform.system()} {platform.release()}",
            "=" * 60,
            ""
        ]

        # Group by status
        errors = [r for r in self.results if r.status == "ERROR"]
        warnings = [r for r in self.results if r.status == "WARNING"]
        ok = [r for r in self.results if r.status == "OK"]

        if errors:
            lines.append("❌ ERRORS (Critical):")
            for r in errors:
                lines.append(f"  • {r.component}: {r.message}")
            lines.append("")

        if warnings:
            lines.append("⚠️  WARNINGS:")
            for r in warnings:
                lines.append(f"  • {r.component}: {r.message}")
            lines.append("")

        if ok:
            lines.append("✅ PASSED:")
            for r in ok:
                lines.append(f"  • {r.component}: {r.message}")
            lines.append("")

        # Summary
        lines.append("=" * 60)
        lines.append(f"Summary: {len(ok)} OK, {len(warnings)} Warnings, {len(errors)} Errors")

        if errors:
            lines.append("Status: CRITICAL - Cannot start")
        elif warnings:
            lines.append("Status: DEGRADED - Can start with limitations")
        else:
            lines.append("Status: HEALTHY - All systems operational")

        lines.append("=" * 60)

        return "\n".join(lines)


def run_health_check() -> bool:
    """
    Run system health check and return True if all critical checks pass.

    Returns:
        bool: True if system is healthy enough to start
    """
    checker = SystemHealthChecker()
    all_passed, results = checker.check_all()

    report = checker.generate_report()
    print(report)
    logger.info("Health check completed")

    return all_passed


if __name__ == "__main__":
    # Run health check from command line
    success = run_health_check()
    sys.exit(0 if success else 1)

