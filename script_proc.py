'''Using the stored JSON, CSV files, run to generate processed data for this study'''

import process
import utilities as utils

#-----------------------------------------------------------------------------------------------------------------------
# GENERATE LIST OF SUBJECTS
#-----------------------------------------------------------------------------------------------------------------------
subjects = utils.generate_subject_list()

"""
#-----------------------------------------------------------------------------------------------------------------------
# IMPORT DATA FOR EACH SUBJECT
#-----------------------------------------------------------------------------------------------------------------------
subjects_data = process.import_process_signals(subjects)

#-----------------------------------------------------------------------------------------------------------------------
# PROCESS DATA FOR EACH SUBJECT
#-----------------------------------------------------------------------------------------------------------------------
process.write_signals_to_json(subjects_data)
process.write_sub_info_to_csv(subjects, subjects_data)
process.write_signals_to_csv(subjects, subjects_data)
process.write_times_to_csv(subjects)
"""

#-----------------------------------------------------------------------------------------------------------------------
# PLOT AND ANALYSE SIGNALS USING PROCESSED JSON FILE
#-----------------------------------------------------------------------------------------------------------------------
# use stored json file
process.plot_emg_torque(subjects, plot_fig=True)
slopes = process.plot_normalised_emg_calc_slopes(subjects, plot_fig=True)
process.analyse_slopes_plot_hist(subjects, slopes, plot_fig=True)
process.write_slopes_to_csv(subjects, slopes)
process.get_mean_normalised_emg_at_activation_levels()
process.analyse_times(plot_fig=True)

