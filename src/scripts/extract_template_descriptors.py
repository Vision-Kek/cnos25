import hydra

@hydra.main(version_base=None, config_path="../../configs", config_name="extract_templates")
def compute_and_save_reference_descriptors(cfg):
    templateextration = hydra.utils.instantiate(cfg.descriptor_extraction)

    templateextration.set_ref_images()
    templateextration.calc_ref_embs()
    templateextration.save(cfg.out_file, overwrite=False)
    templateextration.check_saved(cfg.out_file)


if __name__ == "__main__":
    compute_and_save_reference_descriptors()