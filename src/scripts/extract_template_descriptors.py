import hydra

@hydra.main(version_base=None, config_path="../../configs", config_name="extract_templates")
def compute_and_save_reference_descriptors(cfg):
    templateextration = hydra.utils.instantiate(cfg.descriptor_extraction)

    # Extract the cropped and masked template images for each object
    templateextration.set_ref_images()
    # Forward pass through the descriptor model to get descriptors
    templateextration.calc_ref_embs()

    out_file = cfg.out_file if cfg.out_file is not None else cfg.descriptor_extraction.descriptor_model.cfg.cache_file
    templateextration.save(out_file, overwrite=cfg.overwrite)
    templateextration.check_saved(out_file)


if __name__ == "__main__":
    compute_and_save_reference_descriptors()