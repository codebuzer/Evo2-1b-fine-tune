FROM nvcr.io/nvidia/clara/bionemo-framework:nightly

WORKDIR /app

# Install dependencies
RUN pip install pandas scikit-learn

# Create app folders
RUN mkdir -p /app/data /app/scripts /app/preprocessed_data /app/covid_evo2_model

# Copy scripts
COPY scripts/ /app/scripts/
COPY run.sh /app/
RUN chmod +x /app/run.sh
