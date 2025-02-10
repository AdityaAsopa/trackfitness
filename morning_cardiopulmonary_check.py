import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal, ks_2samp
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
import logging
from datetime import datetime

@dataclass
class BreathingMetrics:
    """Store breathing analysis results"""
    num_cycles: int
    respiratory_rate: float
    cycle_time_mean: float
    cycle_time_std: float
    breathing_delta_hr: float
    inhalation_time: float
    exhalation_time: float
    inhalation_fraction: float
    cycles_interp: List[np.ndarray]
    hr_peaks: np.ndarray
    hr_troughs: np.ndarray
    raw_data: List[np.ndarray]

class OrthostaticAnalyzer:
    """Analyze orthostatic heart rate variability measurements"""
    
    def __init__(self, data_path: Path, origin: str = 'Polar Sensor Logger Export'):
        self.data_path = Path(data_path)
        self.recording_type = Path(data_path).stem.split('_')[-1]
        self.origin = origin
        self.folder = data_path.parent
        self.df = self._load_data()
        self.record_datetime = pd.to_datetime(self.df['Timestamp'].iloc[0]).strftime('%Y-%m-%d %H-%M-%S')
        self.record_date = pd.to_datetime(self.df['Timestamp'].iloc[0]).strftime('%Y-%m-%d')
        self._preprocess_data()
        self.analyze_breathing()
        # self.breath_metrics_raw = breathing_raw
        # self.breath_metrics_baseline_corrected = breahting_baseline_corrected
        self.hr_metrics = self.analyze_hr_recovery()
        
    def _load_data(self) -> pd.DataFrame:
        """Load and parse workout data from various file formats"""
        if 'Polar Sensor Logger Export' in self.origin:
            if self.recording_type == 'HR' or self.recording_type == 'RR':
                df = pd.read_csv(self.data_path, sep=';')
                df = df.rename(columns={
                    'Phone timestamp': 'Timestamp',
                    'HR [bpm]': 'HR',
                    'RR-interval [ms]': 'RR'
                })
                if 'HR' not in df.columns:
                    df['HR'] = 60000 / df['RR']
                elif 'RR' not in df.columns:
                    df['RR'] = 60000 / df['HR']

            elif self.recording_type == 'ECG':
                # columns in csv are: Phone timestamp;sensor timestamp [ns];timestamp [ms];ecg [uV]
                df = pd.read_csv(self.data_path, sep=';')
                df = df.drop(columns=['sensor timestamp [ns]','Phone timestamp'])
                df = df.rename(columns={
                    'timestamp [ms]': 'Timestamp',
                    'ecg [uV]': 'ECG'
                })
                # convert ecg into rr intervals
                df = self._ecg_to_rr(df)
                
        elif 'Runalyze' in self.origin:
            if 'Fit' in self.origin:
                df = self._parse_fit_file()
            else:
                df = self._parse_tcx_file()
                
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        return df
    
    def _ecg_to_rr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert ECG data to RR intervals"""
        # find peaks
        peaks, props = find_peaks(df['ECG'], height=500, prominence=700)
        df2 = pd.DataFrame()
        df2['Timestamp'] = df['Timestamp'].iloc[peaks]
        df2['RR'] = np.array([0, *np.diff(df2['Timestamp'])])
        df2 = df2[(df2['RR'] > 500) & (df2['RR'] < 1500)]
        df2['HR'] = 60000 / df2['RR']
        return df2
    
    def _preprocess_data(self):
        """Prepare data for analysis by calculating epochs and normalizing time"""
        self.df['Timestamp'] = (self.df['Timestamp'] - self.df['Timestamp'].iloc[0]).dt.total_seconds()
        max_hr_time = self.df['Timestamp'][self.df['HR'].idxmax()]
        end_time = self.df['Timestamp'].iloc[-1]
        
        self.pre_baseline = max_hr_time - 30
        self.post_baseline = end_time - (max_hr_time + 70)
        self.recording_length = end_time
        
        self.df['epoch'] = pd.cut(
            self.df['Timestamp'],
            bins=[-np.inf, self.pre_baseline, end_time-self.post_baseline, np.inf],
            labels=['pre', 'orthostatic', 'post']
        )
    
    def _interpolate_cycle(self, cycle: np.ndarray, num_points: int = 15) -> np.ndarray:
        """
        Interpolate a breathing cycle to a fixed number of points.
        
        Args:
            cycle: Array of heart rate values for one breathing cycle
            num_points: Number of points to interpolate to
            
        Returns:
            np.ndarray: Interpolated cycle with uniform sampling
        """
        x = np.linspace(0, 1, len(cycle))
        x_new = np.linspace(0, 1, num_points)
        interpolator = interp1d(x, cycle, kind='cubic')
        return interpolator(x_new)

    def extract_breath_cycles(self, hr: np.ndarray, t: np.ndarray, baseline:np.ndarray) -> BreathingMetrics:
        """
        Extract inhalation and exhalation cycles from heart rate data.
        
        Args:
            hr: Array of heart rate values
            t: Array of timestamp values
        """
        min_normal_breathing_time = 10 #seconds, or 6 cycles per min
        expected_breath_cycles = 0.66 * len(hr) /  min_normal_breathing_time # for 3 min data: 180s / 10s = 18 cycles, 0.66 is for 2/3 of the min breathing rate
        
        # Find peaks and troughs
        min_fluctuation= 2
        maxima_indices, _ = find_peaks(hr, prominence=min_fluctuation)
        minima_indices, _ = find_peaks(-hr)
        
        # Calculate breathing cycle parameters
        num_cycles = min(len(maxima_indices), len(minima_indices))
        cycle_times = np.diff(t[maxima_indices])

        if num_cycles < expected_breath_cycles:
            logging.warning(f"Breathing patterns not detected, expected: {expected_breath_cycles}, found: {len(maxima_indices)}")
            breathing_metrics= BreathingMetrics(
                                                num_cycles,
                                                np.nan,
                                                np.nan,
                                                np.nan,
                                                np.percentile(hr, 95) - np.percentile(hr, 5),
                                                np.nan,
                                                np.nan,
                                                np.nan,
                                                [],
                                                [],
                                                [],
                                                [t,hr,np.zeros_like(hr)]
                                            )
            return breathing_metrics
        
        # Extract and interpolate HR cycles
        hr_cycles = [hr[m1:m2] for m1, m2 in zip(maxima_indices[:-1], maxima_indices[1:])]
        hr_cycles_interp = [self._interpolate_cycle(cycle) for cycle in hr_cycles if len(cycle) > 5]

        if maxima_indices[0] < minima_indices[0]:
            inhalation_times = [round(mint - maxt, 2) for maxt, mint in zip(t[maxima_indices], t[minima_indices])]
            exhalation_times = [round(maxt - mint, 2) for maxt, mint in zip(t[maxima_indices[1:]], t[minima_indices])]
        else:
            inhalation_times = [round(mint - maxt, 2) for maxt, mint in zip(t[maxima_indices], t[minima_indices[1:]])]
            exhalation_times = [round(maxt - mint, 2) for maxt, mint in zip(t[maxima_indices], t[minima_indices])]
        
        mean_inhalation = np.abs(np.mean(inhalation_times))
        mean_exhalation = np.abs(np.mean(exhalation_times))
        inhalation_fraction = round(mean_inhalation / (mean_inhalation + mean_exhalation), 2)
        

        breathing_metrics = BreathingMetrics(
            int(num_cycles),
            60 / np.mean(cycle_times),
            np.mean(cycle_times),
            np.std(cycle_times),
            np.mean(hr[maxima_indices][:num_cycles] - hr[minima_indices][:num_cycles]),
            mean_inhalation,
            mean_exhalation,
            inhalation_fraction,
            hr_cycles_interp,
            maxima_indices,
            minima_indices,
            [t,hr,baseline]
        )

        return breathing_metrics
    
    def analyze_breathing(self) -> BreathingMetrics:
        """
        Analyze breathing patterns from heart rate data.
        
        Returns:
            BreathingMetrics: Object containing breathing analysis results
            
        Raises:
            ValueError: If breathing patterns cannot be reliably detected
        """
        # Extract pre-exercise heart rate data
        hr = self.df[self.df['epoch']=='pre']['HR'].values
        t = self.df[self.df['epoch']=='pre']['Timestamp'].values
        
        window = 30
        # first pad the data with the first value repeated window times
        paddedhr = np.pad(hr, (window, window), 'edge')
        paddedt = np.pad(t, (window, window), 'edge')
        baseline = np.convolve(paddedhr, np.ones(2*window), 'same') / (2*window)

        hr_filtered = (paddedhr - baseline)[window:-window]
        t_filtered = paddedt[window:-window]
        
        # Extract breathing cycles for both raw and baseline corrected HR data
        self.breath_metrics_raw = self.extract_breath_cycles(hr, t, np.zeros_like(hr))
        self.breath_metrics_baseline_corrected = self.extract_breath_cycles(hr_filtered, t_filtered, baseline[window:-window])
        
        # Generate visualization if breathing patterns are detected
        if self.breath_metrics_baseline_corrected.respiratory_rate is not np.nan:
            self._plot_breathing_analysis()
        
        return None

    def _plot_breathing_analysis(self) -> None:
        """
        Create visualization of breathing analysis results.
        
        Args:
            timestamps: Array of timestamp values
            heart_rates: Array of filtered heart rate values
            metrics: BreathingMetrics object containing analysis results
        """
        # Create a blank figure
        fig = plt.figure(figsize=(20,20), layout="tight")
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0,0]) # 1st subplot in a 2x2 grid
        ax2 = fig.add_subplot(gs[0,1], projection='polar') # 2nd subplot in a 2x2 grid
        ax3 = fig.add_subplot(gs[1,0], ) # 3rd subplot in a 2x2 grid
        ax4 = fig.add_subplot(gs[1,1], projection='polar') # 4th subplot in a 2x2 grid
        ax5 = fig.add_subplot(gs[2,:]) # 5th subplot spanning the whole bottom row

        hrcycles = self.breath_metrics_raw.cycles_interp
        self._plot_breath_cycles(hrcycles, ax1, kind='rect',  ylim=[40,80])
        self._plot_breath_cycles(hrcycles, ax2, kind='polar', ylim=[40,80])

        hrcycles = self.breath_metrics_baseline_corrected.cycles_interp
        self._plot_breath_cycles(hrcycles, ax3, kind='rect',  ylim=[-15,15])
        self._plot_breath_cycles(hrcycles, ax4, kind='polar', ylim=[-15,15])

        # last axes for plotting raw hr and baseline values
        t,hr,_ = self.breath_metrics_raw.raw_data
        ax5.plot(t, hr, 'b', lw=3, label='Raw HR')
        t,hr,baseline = self.breath_metrics_baseline_corrected.raw_data
        ax5.plot(t, hr, 'purple', lw=3, label='Baseline corrected HR')
        ax5.plot(t, baseline, 'grey', label='Baseline')
        # add hrpeaks and troughs
        hr_peaks, hr_troughs = self.breath_metrics_baseline_corrected.hr_peaks, self.breath_metrics_baseline_corrected.hr_troughs
        ax5.scatter(t[hr_peaks], hr[hr_peaks], s=100, color='r', marker='o', label='HR Peaks', zorder=10)
        ax5.scatter(t[hr_troughs], hr[hr_troughs], s=100, color='g', marker='o', label='HR Troughs', zorder=10)
        # remove spines
        [ax5.spines[x].set_visible(False) for x in ['top', 'right']]
        ax5.legend()
        ax5.set_ylim([-20,100])

        # add text as titles in the middle of the figure
        fig.text(0.5, 0.98, 'Raw HR Breathing Analysis', ha='center', va='center', fontsize=20)
        fig.text(0.5, 0.65, 'Baseline Corrected HR Breathing Analysis', ha='center', va='center', fontsize=20)
        fig.text(0.5, 0.30, 'HR and Baseline Values', ha='center', va='center', fontsize=20)

        # Save plot
        plt.savefig(self.folder / f'breathing_analysis_{self.record_datetime}.png')
        plt.close()

    def _plot_breath_cycles(self, cycles, ax, kind='rect', ylim=[40,80], ):
        """Plot a single breathing cycle"""
        if kind=='rect':
            dlim = 180
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        elif kind=='polar':
            dlim = np.pi
            ax.set_theta_zero_location("N")
            ax.spines['polar'].set_visible(False)
        
        t_cycle = np.linspace(0, 2*dlim, len(cycles[0]))
        # shade inhalation and exhalation phases
        lower_limit, upper_limit = ylim
        
        # # # Shade the inhalation phase
        ax.fill_between(np.linspace(0,dlim,180), np.linspace(lower_limit,lower_limit,180), np.linspace(upper_limit,upper_limit,180), color=np.array([37,134,142])/255, alpha=0.5)
        # # # Shade the exhalation phase
        ax.fill_between(np.linspace(dlim,2*dlim,180), np.linspace(lower_limit,lower_limit,180), np.linspace(upper_limit,upper_limit,180), color=np.array([164,49,127])/255, alpha=0.5)

        [ax.plot(t_cycle, cycle, 'grey') for cycle in cycles]
        # plot avg
        ax.plot(t_cycle, np.mean(cycles, axis=0), color='k', linewidth=2, label='Average Cycle')
        ax.set_xlabel('Time')
        ax.set_ylabel('Heart Rate')
        ax.set_title('Breathing Cycle')
        ax.yaxis.grid(color='white')
        ax.xaxis.grid(color='white')
        # x-axis ticks should be at 0, dlim/4, dlim/2, 3dlim/4, dlim
        ax.set_xticks([0, dlim/4, dlim/2, 3*dlim/4, dlim, 5*dlim/4, 3*dlim/2, 7*dlim/4, 2*dlim])
        ax.set_yticks(np.linspace(ylim[0],ylim[1],7))        
        ax.set_ylim(ylim)

        ax.legend()

        return ax
    
    @staticmethod
    def calculate_hrv_metrics(rr_intervals: np.ndarray) -> Tuple[float, float]:
        """Calculate RMSSD and SDNN from RR intervals"""
        rmssd = np.sqrt(np.median(np.diff(rr_intervals)**2))
        sdnn = np.std(rr_intervals)
        return round(rmssd, 2), round(sdnn, 2)
       
    def analyze_hr_recovery(self) -> Dict[str, float]:
        """
        Analyze heart rate recovery patterns by calculating:
        1. HR rise slope before peak in orthostatic epoch
        2. HR value 60 seconds after peak in orthostatic epoch
        """
        # pre period analysis
        per_epoch_median_hr = self.df[self.df['epoch']=='pre']['HR'].median()
        pre_epoch_hr_std    = self.df[self.df['epoch']=='pre']['HR'].std()

        # Get orthostatic epoch data
        ortho_data = self.df[self.df['epoch'] == 'orthostatic']
        ortho_epoch_start_timepoint = ortho_data['Timestamp'].iloc[0]
        
        # Find most prominent peak
        max_hr = ortho_data['HR'].max()
        peaks, peak_props = find_peaks(ortho_data['HR'].values, height=max_hr-5, width=5)
        peak_idx = np.argmax(peak_props['prominences']) # out of all detected peaks, index of the most prominent one
        peak_start = peak_props['left_bases'][peak_idx]
        peak_point = peaks[peak_idx]
        
        # Calculate rise metrics
        ortho_epoch_hrrise_start_timepoint = ortho_data['Timestamp'].iloc[peak_start] 
        ortho_epoch_peak_timepoint = ortho_data['Timestamp'].iloc[peak_point]
        rise_segment = ortho_data.iloc[peak_start:peak_point]
        hr_rise = rise_segment['HR'].iloc[-1] - rise_segment['HR'].iloc[0]
        rise_time = rise_segment['Timestamp'].iloc[-1] - rise_segment['Timestamp'].iloc[0]
        rise_slope = hr_rise / rise_time
        
        # Calculate fall metrics
        median_hr_60s = ortho_data['HR'].iloc[-20:].median()  # Last 20s median (from 50s to 70s after hr peak)
        HRR60 = max_hr - median_hr_60s

        # post period analysis
        post_epoch_median_hr = self.df[self.df['epoch']=='post']['HR'].median()
        post_epoch_hr_std    = self.df[self.df['epoch']=='post']['HR'].std()

        #post - pre comparison
        pre_to_post_hr_load = post_epoch_median_hr - per_epoch_median_hr

        # Return metrics
        return {
                "pre_epoch_median_hr": per_epoch_median_hr,
                "pre_epoch_hr_std": pre_epoch_hr_std,
                "ortho_epoch_start_timepoint": ortho_epoch_start_timepoint,
                "ortho_epoch_hrrise_start_timepoint": ortho_epoch_hrrise_start_timepoint,
                "ortho_epoch_peak_timepoint": ortho_epoch_peak_timepoint,
                "ortho_hr_initial": rise_segment['HR'].iloc[0],
                "ortho_hr_peak": max_hr,
                "hr_rise": hr_rise,
                "rise_slope": rise_slope,
                "orthostatic_load": max_hr - per_epoch_median_hr,
                "HRR60": HRR60,
                "post_epoch_median_hr": post_epoch_median_hr,
                "post_epoch_hr_std": post_epoch_hr_std,
                "pre_to_post_hr_load": pre_to_post_hr_load
            }

    def assess_condition(self) -> Tuple[str, str]:
        """Assess overall condition based on HR patterns"""
        pre_hr = self.df[self.df['epoch']=='pre']['HR']
        post_hr = self.df[self.df['epoch']=='post']['HR']
        
        # Statistical comparison
        _, ks_pval = ks_2samp(pre_hr, post_hr, alternative='greater')
        median_diff = post_hr.median() - pre_hr.median()
        
        # Determine condition
        condition = 'Fit' if (ks_pval < 0.001 and median_diff > 5) else 'Recover'
        
        # Check for anomalies
        max_ortho_hr = self.df[self.df['epoch']=='orthostatic']['HR'].max()
        max_post_hr = post_hr.max()
        max_pre_hr = pre_hr.max()
        
        hr_flag = 'None'
        if max_ortho_hr < max_post_hr:
            hr_flag = 'post_High'
        elif max_ortho_hr < max_pre_hr:
            hr_flag = 'pre_High'

        # Check for breathing anomalies
        breathing_flag = 'None'
        # if self.breathing.breathing_delta_hr is less than 5, then breathing_flag = 'Shallow Breathing'
        if self.breath_metrics_baseline_corrected.breathing_delta_hr < 5:
            breathing_flag = 'Shallow_Breathing'
        elif self.breath_metrics_baseline_corrected.breathing_delta_hr > 10:
            breathing_flag = 'Deep_Breathing'

        data_flag = 'breathing_not_detected' if self.breath_metrics_baseline_corrected.respiratory_rate is np.nan else 'breathing_detected'
            
        return condition, hr_flag, breathing_flag, data_flag
    
    def generate_report(self) -> Dict[str, float]:
        """Generate comprehensive analysis report"""
        pre_hrv     = self.calculate_hrv_metrics(self.df[self.df['epoch']=='pre']['RR'])
        hr_recovery = self.hr_metrics
        breathing   = self.breath_metrics_baseline_corrected
        condition, hr_flag, breathing_flag, data_flag = self.assess_condition()
        if breathing is None:
            breathing_data = {
                'num_cycles': np.nan,
                'respiratory_rate': np.nan,
                'cycle_time_mean': np.nan,
                'cycle_time_std': np.nan,
                'breathing_delta_hr': np.nan,
                'inhalation_time': np.nan,
                'exhalation_time': np.nan,
                'inhalation_fraction': np.nan
            }
        else:
            breathing_data = {
                'num_cycles': breathing.num_cycles,
                'respiratory_rate': breathing.respiratory_rate,
                'cycle_time_mean': breathing.cycle_time_mean,
                'cycle_time_std': breathing.cycle_time_std,
                'breathing_delta_hr': breathing.breathing_delta_hr,
                'inhalation_time': breathing.inhalation_time,
                'exhalation_time': breathing.exhalation_time,
                'inhalation_fraction': breathing.inhalation_fraction
            }
        
        report = {
            'date': self.record_date,
            'timestamp': self.record_datetime,
            'recording_type':self.recording_type,
            'recording_length':self.recording_length,
            'pre_baseline' : self.pre_baseline,
            'post_baseline' : self.post_baseline,
            'condition': condition,
            'pre_hrv_rmssd': pre_hrv[0],
            'pre_hrv_sdnn':  pre_hrv[1],
            'hr_flag': hr_flag,
            'breathing_flag': breathing_flag,
            'data_flag': data_flag,
            **hr_recovery,
            **breathing_data  # Unpack breathing metrics
        }
            
        return report
    
    def save_to_csv(self, report: Dict[str, float], filename: str = "orthostatic_hrv_record_2025_v2.csv") -> None:
        """Save analysis results to CSV file"""
        csv_path = self.folder / filename
        
        # Convert report to DataFrame
        report_df = pd.DataFrame([report], index=[0])
        
        # If file exists, append; if not, create new file
        dfsave = pd.read_csv(csv_path) if Path(csv_path).exists() else pd.DataFrame()
        dfsave = pd.concat([report_df, dfsave], ignore_index=True)

        dfsave.to_csv(csv_path, index=False)
        logging.info(f"Results saved to {csv_path}")
    
    def plot_analysis(self) -> None:
        """Generate visualization of the analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot HR time series
        sns.lineplot(data=self.df, x='Timestamp', y='HR', hue='epoch', ax=ax1)
        ax1.set_title('Heart Rate Response')
        
        # Plot HR recovery
        recovery = self.hr_metrics
        ortho_data = self.df[self.df['epoch']=='orthostatic']
        ax2.plot(ortho_data['Timestamp'], ortho_data['HR'])

        t0, t1          = recovery['ortho_epoch_hrrise_start_timepoint'], recovery['ortho_epoch_peak_timepoint']
        hr0, hr1, hr2   = recovery['ortho_hr_initial'], recovery['ortho_hr_peak'], recovery['HRR60']
        ax2.plot([t0,t1],[hr0, hr1], '-o', color='orange',  label=f'HR Rise: {recovery["rise_slope"]:.1f} bpm/s')
        ax2.plot([t1,t1+60],[hr1,hr1-hr2], '-o', color='green', label=f'60s HR Recovery: {hr2:.1f} bpm')
        
        ax2.fill_between(ortho_data['Timestamp'], recovery['pre_epoch_median_hr']-2*recovery['pre_epoch_hr_std'], recovery['pre_epoch_median_hr']+2*recovery['pre_epoch_hr_std'], color='blue', alpha=0.2)
        ax2.axhline(y=recovery['pre_epoch_median_hr'], color='blue', linestyle='--', label='pre epoch median HR')
        
        ax2.axhline(y=hr1, color='r', linestyle='--', label='HR Peak')
        ax2.axhline(y=hr1-hr2, color='purple', linestyle='--', label='HR after 60s')
        
        ax2.fill_between(ortho_data['Timestamp'], recovery['post_epoch_median_hr']-2*recovery['post_epoch_hr_std'], recovery['post_epoch_median_hr']+2*recovery['post_epoch_hr_std'], color='green', alpha=0.2)
        ax2.axhline(y=recovery['post_epoch_median_hr'], color='green', linestyle='--', label='post epoch median HR')
        
        ax2.set_title('Ortho Epoch')
        ax2.set_ylim([40, 120])
        ax2.legend()

        plt.tight_layout()
        plt.savefig(self.folder / f'orthostatic_analysis_{self.record_datetime}.png')
        plt.close()

def main(ecg_data_filepath, origin='Polar Sensor Logger Export'):
    logging.basicConfig(level=logging.INFO)
       
    analyzer = OrthostaticAnalyzer(ecg_data_filepath, origin=origin)
    report = analyzer.generate_report()
    analyzer.plot_analysis()
    
    # Save results to CSV
    analyzer.save_to_csv(report)
    return report

if __name__ == "__main__":
    ecg_data_filepath = Path(sys.argv[1])
    report = main(ecg_data_filepath,)
    logging.info("Analysis Results:")
    for metric, value in report.items():
        logging.info(f"{metric}: {value}")
