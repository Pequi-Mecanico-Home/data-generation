from sdg_engine.core.interfaces.blender.scene import BlenderScene
from sdg_engine.core.interfaces.blender.sweep import BlenderSweep
from sdg_engine.core.interfaces.blender.object import BlenderElement
from sdg_engine.config import RenderingConfig # Certifique-se que RenderingConfig tem o campo background_images_folder_path
from sdg_engine.core.interfaces.blender import utils

from sdg_engine.core.model import Dataset, Annotation, SnapshotAnnotation

import bpy
from tqdm import tqdm
from typing import List, Tuple, Optional # Adicionado Optional para os novos parâmetros
import uuid
import warnings
import numpy as np
import os
import glob
import random # Adicionado para seleção aleatória, embora o loop seja sequencial agora

METADATA_FILENAME = "metadata.jsonl"

class BlenderRenderer:
    """Interface for Blender rendering."""

    def __init__(
        self,
        scene: BlenderScene,
        target_path: str,
        resolution: Tuple[int, int],
        samples: int,
    ):
        """Initialize the interface with a Blender scene.

        Parameters:
        ___________
        scene: BlenderScene
            The Blender scene to render.
        target_path: str
            The path to save the rendered snapshots.
        resolution: Tuple[int, int]
            The resolution of the rendered snapshots.
        samples: int
            The number of samples to use for the rendered snapshots.
        """
        self.scene = scene
        self.target_path = target_path
        self.resolution = resolution
        # Set blender scene settings
        self.scene.blender_scene.render.resolution_x = resolution[0]
        self.scene.blender_scene.render.resolution_y = resolution[1]
        self.scene.blender_scene.cycles.samples = samples
        # A resolução percentual e a transparência do filme são manipuladas em scene.py agora.
        # self.scene.blender_scene.render.resolution_percentage = 100
        # self.scene.blender_scene.render.film_transparent = False


    @classmethod
    def from_scene(
        cls,
        scene: BlenderScene,
        target_path: str,
        resolution: Tuple[int, int],
        samples: int,
    ):
        """Create a BlenderRenderer object from an existing scene."""
        return cls(scene, target_path, resolution, samples)

    def render_snapshot(
        self,
        snapshot_id: Optional[uuid.UUID] = None, # Torna opcional, para compatibilidade
        custom_filepath: Optional[str] = None # NOVO: Caminho de arquivo completo customizado
    ) -> str:
        """Render a snapshot of the scene and return the path to the snapshot."""
        if custom_filepath:
            self.scene.blender_scene.render.filepath = custom_filepath
        elif snapshot_id:
            self.scene.blender_scene.render.filepath = f"{self.target_path}/{snapshot_id}.png"
        else:
            raise ValueError("Either snapshot_id or custom_filepath must be provided for rendering.")
            
        bpy.ops.render.render(write_still=True)
        return self.scene.blender_scene.render.filepath


    def annotate_snapshot(
        self,
        cameras: List[BlenderElement],
        elements: List[BlenderElement],
        snapshot_id: uuid.UUID, # Mantém como UUID, representa o "snapshot" lógico base
        relative: bool = True,
        file_name_override: Optional[str] = None # NOVO: Nome do arquivo final da imagem
    ) -> Annotation:
        """Create bounding boxes for the elements in the scene."""
        if len(cameras) > 1:
            warnings.warn(
                "Multiple cameras are not supported yet. Only the first camera will be used."
            )
        camera = cameras[0]

        # Usa file_name_override se fornecido, senão usa o padrão baseado no snapshot_id
        annotation = Annotation(
            file_name=file_name_override if file_name_override else f"{snapshot_id}.png",
            objects=SnapshotAnnotation(bbox=[], categories=[]),
        )

        # --- Obtém a resolução atual da cena do Blender para o cálculo da bounding box ---
        current_render_resolution = (
            self.scene.blender_scene.render.resolution_x,
            self.scene.blender_scene.render.resolution_y
        )
        # --- FIM MUDANÇA ---

        for i, element in enumerate(elements):
            bounding_box: np.ndarray = utils.create_bounding_box(
                scene=self.scene,
                camera=camera,
                element=element,
                relative=relative,
                resolution=current_render_resolution, # Usa a resolução ATUAL da cena
            )
            if bounding_box is None:
                continue

            annotation.objects.bbox.append(bounding_box.tolist())
            annotation.objects.categories.append(i)

        return annotation


def generate_dataset_from_config(config: RenderingConfig) -> Dataset:
    """Generate a dataset from a rendering configuration."""
    # Initialize the scene and sweep
    # A cena é carregada e os nós do mundo são configurados em BlenderScene.from_scene_config
    scene: BlenderScene = BlenderScene.from_scene_config(config.scene_config)
    sweep: BlenderSweep = BlenderSweep.from_sweep_config(config.sweep_config)

    # Initialize the renderer
    split_path = f"{config.target_path}/{config.split}"
    renderer: BlenderRenderer = BlenderRenderer.from_scene(
        scene,
        split_path,
        config.resolution,
        config.samples,
    )

    # Initialize the dataset
    dataset: Dataset = Dataset(path=split_path, annotations=[])

    # --- INÍCIO: NOVAS MUDANÇAS PARA MÚLTIPLOS BACKGROUNDS ---

    # 1. Obter o caminho da pasta de backgrounds da configuração
    # Este assume que 'background_images_folder_path' foi adicionado a SceneConfig em config.py
    background_images_folder_path = getattr(config.scene_config, 'background_images_folder_path', None)
    all_background_images = []

    if background_images_folder_path and os.path.isdir(background_images_folder_path):
        image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp')
        for ext in image_extensions:
            all_background_images.extend(glob.glob(os.path.join(background_images_folder_path, ext)))
        
        if not all_background_images:
            warnings.warn(f"Nenhuma imagem de background encontrada em: '{background_images_folder_path}'. As renderizações terão um fundo de cor sólida padrão.")
    elif background_images_folder_path: # Se o caminho foi dado, mas não é um diretório
        warnings.warn(f"O caminho fornecido para backgrounds não é um diretório válido: '{background_images_folder_path}'. As renderizações terão um fundo de cor sólida padrão.")
    else: # Se nenhum caminho foi fornecido
        warnings.warn("Nenhum caminho para pasta de backgrounds fornecido em config.scene_config.background_images_folder_path. As renderizações terão um fundo de cor sólida padrão.")
    
    # Se nenhuma imagem foi encontrada ou o caminho não é válido, a lista all_background_images estará vazia
    # E isso ativará o fallback para cor sólida.

    # --- FIM: NOVAS MUDANÇAS PARA MÚLTIPLOS BACKGROUNDS ---

    # Collect the dataset annotations
    # O loop principal agora será aninhado
    for snapshot_idx, snapshot in enumerate(tqdm(sweep.snapshots, desc="Rendering snapshots")):
        # Prepare axis, camera and light para o snapshot atual
        scene.prepare_from_snapshot(snapshot=snapshot)

        # --- NOVO LOOP PARA CADA BACKGROUND ---
        # Determina quais backgrounds usar. Se houver backgrounds válidos, usa-os.
        # Caso contrário, usa uma lista com um 'fundo padrão' para garantir que o loop interno execute pelo menos uma vez.
        backgrounds_to_render = all_background_images if all_background_images else [None]

        for bg_idx, background_filepath in enumerate(backgrounds_to_render):
            if background_filepath:
                print(f"Applying background: {os.path.basename(background_filepath)}")
                scene.set_background_image(background_filepath)
                # A resolução da cena será ajustada por set_background_image se a imagem for carregada com sucesso
            else:
                # Se background_filepath for None (no caso de fallback), usa cor sólida
                scene.set_solid_background_color()
                
                # Quando usamos cor sólida, garantimos que a resolução é a definida na config
                scene.blender_scene.render.resolution_x = renderer.resolution[0]
                scene.blender_scene.render.resolution_y = renderer.resolution[1]
                scene.blender_scene.render.resolution_percentage = 100

            # --- RENDERIZAÇÃO E ANOTAÇÃO ---
            # Crie um nome de arquivo único para a renderização atual (snapshot + background)
            # Usamos snapshot.id.hex para uma representação compacta e válida no nome do arquivo.
            render_file_base_name = f"{snapshot.id.hex}_{bg_idx:04d}" 
            full_output_filepath = f"{renderer.target_path}/{render_file_base_name}.png"
            
            renderer.render_snapshot(custom_filepath=full_output_filepath)
            
            annotation: Annotation = renderer.annotate_snapshot(
                cameras=scene.cameras,
                elements=scene.elements,
                snapshot_id=snapshot.id, # Passa o UUID original do snapshot base para a anotação
                file_name_override=f"{render_file_base_name}.png" # Nome do arquivo que foi realmente renderizado
            )

            if config.debug:
                utils.draw_bounding_box_with_category(
                    target_path=split_path,
                    annotation=annotation,
                    snapshot=snapshot, # Snapshot original
                )

            dataset.annotations.append(annotation)
        # --- FIM NOVO LOOP PARA CADA BACKGROUND ---

    # Render the annotation animation
    if config.debug:
        utils.render_annotation_animation(split_path, dataset)

    # Save the dataset to the target path as a JSONL file
    with open(f"{split_path}/{METADATA_FILENAME}", "w") as f:
        for annotation in dataset.annotations:
            f.write(annotation.model_dump_json() + "\n")

    return dataset