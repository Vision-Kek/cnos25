import hydra
import logging

from src.model.descriptor.template_extraction import DinoTemplateExtraction


@hydra.main(version_base=None, config_path="../../configs", config_name="extract_templates")
def compute_and_save_reference_descriptors(cfg):
    # Instantiate the reference dataloader (the one that provides the crops from the onboarding images)
    ref_dataloader = cfg.data.reference_dataloader
    if cfg.dataset_name == 'hot3d':
        ref_dataloader._target_ = 'src.dataloader.bop_hot3d.BOPHOT3DTemplate'
    ref_dataloader = hydra.utils.instantiate(ref_dataloader)

    # Instantiate the object that handles the template descriptor extraction and saving
    template_extraction = DinoTemplateExtraction(descriptor_model=cfg.descriptor_model,
                                                ref_dataloader=ref_dataloader,
                                                obj_names=cfg.data.datasets[cfg.dataset_name].obj_names,
                                                dataset_name=cfg.dataset_name)

    # Extract the cropped and masked template images for each object
    template_extraction.set_ref_images()
    # Forward pass through the descriptor model to get descriptors
    template_extraction.calc_ref_embs()

    # Save descriptors dict to a .pt
    out_file = cfg.out_file if cfg.out_file is not None else cfg.descriptor_model.cfg.cache_file
    template_extraction.save(out_file, overwrite=cfg.overwrite)
    template_extraction.check_saved(out_file)


if __name__ == "__main__":
    compute_and_save_reference_descriptors()