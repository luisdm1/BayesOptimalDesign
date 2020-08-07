from rpy2 import robjects
from rpy2.robjects import r as r_run


def load_zip_csv_samples_file(mcmc_config, path, samples_type, fit_name,
                              file_name, file_zip, zipped=True):
    print(("\n\n>>>>>>>LOADING {} {} CSV SAMPLES FROM ZIP!<<<<<<\n"
           ).format(mcmc_config["MCMC_type"],
                    samples_type
                    )
        )
    if samples_type == "priors":
        suffix_samples_type = "_priors"
    else:
        suffix_samples_type = ""
    if mcmc_config["num_chains"] == 1:
        pattern_csv = file_name + suffix_samples_type + ".csv"
    elif mcmc_config["num_chains"] > 1:
        pattern_csv = file_name + suffix_samples_type + "_[0-9].csv"
    if zipped:
        zipped_str = 'TRUE'
    else:
        zipped_str = 'FALSE'

    try:
        r_run(
            ("""
             filezip = "{}";
             print(c('Unzip zip file: ', filezip))
             path="{}";
             print(c('path is', path))
             zipped = {}
             if (zipped == TRUE){{
                unzip(filezip, exdir=path);
             }}
             pattern = "{}";
             print('pattern:')
             print(pattern);
             csvfiles <- dir(path = path, pattern = pattern, full.names = TRUE);
             print("Loading CSV FILES:")
             print(csvfiles);
             {} <- read_stan_csv(csvfiles);
             print("Sucessfully loaded {} csv samples into R env");
             if (zipped == TRUE){{
                 file.remove(csvfiles)
                 print('Warning: Removed csv files')
             }}else{{
                print('Warning: Did not remove csv files!')}}
             """
            ).format(file_zip, path, zipped_str, pattern_csv, fit_name,
                     samples_type)
            )
        fit = robjects.globalenv[fit_name]
        run_mcmc = False
        print(("\n\n>>>>>>>SUCESSFULLY LOADED {} {} CSV SAMPLES \n{}\n into R"
               + " workspace as {}!<<<<<<\n\n"
              ).format(mcmc_config["MCMC_type"],
                       samples_type,
                       path+pattern_csv,
                       fit_name))
        return fit, run_mcmc
    except:
       print((
           "Warning:File \n {} \n does not exist! \n Running sampling {}..."
           .format(file_zip, samples_type)
           ))
    return None, True
