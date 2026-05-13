"""
Progress tracking module for long-running jobs.
"""

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class ProgressPhase:
    """Represents a single phase of a job."""
    name: str
    weight: float  # Percentage weight of total job
    current_step: int = 0
    total_steps: int = 1
    status: str = 'pending'  # pending, active, completed, error
    message: str = ''
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass 
class JobProgress:
    """Represents the overall progress of a job."""
    job_id: str
    phases: List[ProgressPhase]
    current_phase_index: int = 0
    overall_percent: float = 0.0
    status: str = 'pending'  # pending, running, completed, error
    start_time: float = 0.0
    estimated_total_time: Optional[float] = None
    message: str = ''


class ProgressTracker:
    """Thread-safe progress tracker for managing multiple jobs."""

    def __init__(self):
        self.jobs: Dict[str, JobProgress] = {}
        self.lock = threading.Lock()

    def create_job(self, job_id: str, phases: List[tuple]) -> None:
        """Create a new job with phases. Each phase is (name, weight)."""
        with self.lock:
            progress_phases = [
                ProgressPhase(name=name, weight=weight) 
                for name, weight in phases
            ]
            self.jobs[job_id] = JobProgress(
                job_id=job_id,
                phases=progress_phases,
                start_time=time.time()
            )
            print(f"PROGRESS - Created job {job_id} with {len(phases)} phases")
    
    def start_phase(self, job_id: str, phase_name: str, total_steps: int = 1, 
                   estimated_duration: Optional[float] = None) -> None:
        """Start a specific phase."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                print(f"PROGRESS - Job {job_id} not found")
                return
            
            # Find phase by name
            phase_index = None
            for i, phase in enumerate(job.phases):
                if phase.name == phase_name:
                    phase_index = i
                    break
            
            if phase_index is None:
                print(f"PROGRESS - Phase {phase_name} not found in job {job_id}")
                return
            
            phase = job.phases[phase_index]
            phase.status = 'active'
            phase.total_steps = total_steps
            phase.current_step = 0
            phase.start_time = time.time()
            
            job.current_phase_index = phase_index
            job.status = 'running'
            
            if estimated_duration:
                job.estimated_total_time = estimated_duration
            
            self._update_overall_progress(job_id)
            print(f"PROGRESS - Started phase '{phase_name}' for job {job_id}")
    
    def update_phase(self, job_id: str, phase_name: str, step: int, 
                    message: str = '', substep_progress: float = 0.0) -> None:
        """Update progress within a phase."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            
            # Find phase by name
            phase = None
            for p in job.phases:
                if p.name == phase_name:
                    phase = p
                    break
            
            if not phase:
                return
            
            phase.current_step = step
            phase.message = message
            
            self._update_overall_progress(job_id, substep_progress)
            print(f"PROGRESS - {job_id}/{phase_name}: step {step}/{phase.total_steps} - {message}")
    
    def complete_phase(self, job_id: str, phase_name: str, message: str = '') -> None:
        """Mark a phase as completed."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            
            # Find phase by name
            phase = None
            for p in job.phases:
                if p.name == phase_name:
                    phase = p
                    break
            
            if not phase:
                return
            
            phase.status = 'completed'
            phase.current_step = phase.total_steps
            phase.end_time = time.time()
            if message:
                phase.message = message
            
            self._update_overall_progress(job_id)
            print(f"PROGRESS - Completed phase '{phase_name}' for job {job_id}")
    
    def complete_job(self, job_id: str, message: str = 'Job completed') -> None:
        """Mark entire job as completed."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            
            job.status = 'completed'
            job.overall_percent = 100.0
            job.message = message
            
            # Mark all phases as completed
            for phase in job.phases:
                if phase.status != 'completed':
                    phase.status = 'completed'
                    phase.current_step = phase.total_steps
                    phase.end_time = time.time()
            
            print(f"PROGRESS - Completed job {job_id}")
    
    def error_job(self, job_id: str, error_message: str) -> None:
        """Mark job as failed with error."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            
            job.status = 'error'
            job.message = f'Error: {error_message}'
            
            # Mark current phase as error
            if job.current_phase_index < len(job.phases):
                current_phase = job.phases[job.current_phase_index]
                current_phase.status = 'error'
                current_phase.message = error_message
            
            print(f"PROGRESS - Error in job {job_id}: {error_message}")
    
    def _update_overall_progress(self, job_id: str, substep_progress: float = 0.0) -> None:
        """Calculate overall job progress based on phase weights."""
        job = self.jobs.get(job_id)
        if not job:
            return
        
        total_progress = 0.0
        
        for i, phase in enumerate(job.phases):
            if phase.status == 'completed':
                total_progress += phase.weight
            elif phase.status == 'active':
                # Calculate progress within current phase
                if phase.total_steps > 0:
                    phase_progress = (phase.current_step + substep_progress) / phase.total_steps
                else:
                    phase_progress = substep_progress
                total_progress += phase.weight * phase_progress
        
        job.overall_percent = min(100.0, max(0.0, total_progress))
        
        # Update main message
        if job.current_phase_index < len(job.phases):
            current_phase = job.phases[job.current_phase_index]
            job.message = f"{current_phase.name}: {current_phase.message}"
    
    def get_status(self, job_id: str) -> Optional[Dict]:
        """Get current status of a job."""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            
            # Calculate remaining time if we have estimates
            estimated_remaining = None
            if job.estimated_total_time and job.overall_percent > 0:
                elapsed = time.time() - job.start_time
                if job.overall_percent > 5:  # Only estimate after some progress
                    total_estimated = elapsed * 100 / job.overall_percent
                    estimated_remaining = max(0, total_estimated - elapsed)
            
            return {
                'job_id': job_id,
                'percent': round(job.overall_percent, 1),
                'status': job.status,
                'message': job.message,
                'current_phase': job.phases[job.current_phase_index].name if job.current_phase_index < len(job.phases) else None,
                'estimated_remaining_seconds': round(estimated_remaining) if estimated_remaining else None,
                'phases': [
                    {
                        'name': phase.name,
                        'status': phase.status,
                        'progress': round(phase.current_step / phase.total_steps * 100) if phase.total_steps > 0 else 0,
                        'message': phase.message
                    }
                    for phase in job.phases
                ]
            }


# Global progress tracker instance
PROGRESS_TRACKER = ProgressTracker()


def estimate_download_time(size_bytes: int) -> float:
    """Estimate download time based on file size."""
    size_mb = size_bytes / 1e6
    
    if size_mb < 10:
        return max(5, size_mb * 1.5)
    elif size_mb < 50:
        return size_mb * 1.2
    else:
        return size_mb * 0.8


def estimate_processing_time(size_bytes: int, has_model1: bool = True) -> float:
    """Estimate processing time based on file size and enabled models."""
    size_mb = size_bytes / 1e6
    
    # Base processing time estimates
    base_time = 5  # Minimum processing time
    
    # Index calculation time (models 2-4): ~0.1-0.3 seconds per MB
    index_time = size_mb * 0.2
    
    # Model1 inference time: ~0.5-2 seconds per MB depending on image size
    model1_time = 0
    if has_model1:
        if size_mb < 20:
            model1_time = size_mb * 1.5
        else:
            model1_time = size_mb * 1.0
    
    return base_time + index_time + model1_time

