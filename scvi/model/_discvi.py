import logging
from typing import Any, List, Optional, Tuple

from anndata import AnnData
from scipy.sparse import csr_matrix

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi._decorators import classproperty
from scvi._types import LatentDataType
from scvi.data import AnnDataManager
from scvi.data._constants import _ADATA_LATENT_UNS_KEY
from scvi.data._utils import _get_latent_adata_type
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
    StringUnsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.module import DISVAE
from scvi.module.base import BaseModuleClass
from scvi.utils import setup_anndata_dsp

from .base import ArchesMixin, BaseLatentModeModelClass, RNASeqMixin, VAEMixin

logger = logging.getLogger(__name__)

_SCVI_LATENT_QZM = "_scvi_latent_qzm"
_SCVI_LATENT_QZV = "_scvi_latent_qzv"


logger = logging.getLogger(__name__)


class DISCVI(
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    UnsupervisedTrainingMixin,
    BaseLatentModeModelClass,
):
    """
    Disentangled single-cell Variational Inference.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    disentangling_metric
        One of:

        * ``'hsic'`` - HSIC objective
        * ``'tc'`` - Total correlation objective
    disentangling_weight
        Weight of the disentangling metric.
    kld_weight
        Weight of KL divergence loss.
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`

    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.DISCVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.DISCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()

    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        disentangling_metric: Literal["hsic", "tc"] = "hsic",
        disentangling_weight: float = 1.0,
        kld_weight: float = 1.0,
        **model_kwargs,
    ):
        super().__init__(adata)

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            if self.latent_data_type is not None:
                raise ValueError(
                    "Latent mode not supported when use_size_factor_key is False"
                )

            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_cells=adata.X.shape[0],
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            disentangling_metric=disentangling_metric,
            disentangling_weight=disentangling_weight,
            kld_weight=kld_weight,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            latent_data_type=self.latent_data_type,
            **model_kwargs,
        )
        self._model_summary_string = (
            "SCVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    # Insert DISVAE here, rest is identical to SCVI
    @classproperty
    def _module_cls(cls) -> BaseModuleClass:
        return DISVAE

    @classproperty
    def _tunables(cls) -> Tuple[Any]:
        return (cls._module_cls,)

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.

        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        # register new fields for latent mode if needed
        latent_mode = _get_latent_adata_type(adata)
        if latent_mode is not None:
            anndata_fields += cls._get_latent_fields(latent_mode)
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def _get_latent_adata_from_adata(
        self,
        mode: LatentDataType,
        use_latent_qzm_key: str = "X_latent_qzm",
        use_latent_qzv_key: str = "X_latent_qzv",
    ):
        if mode == "dist":
            self.adata.obsm[_SCVI_LATENT_QZM] = self.adata.obsm[use_latent_qzm_key]
            self.adata.obsm[_SCVI_LATENT_QZV] = self.adata.obsm[use_latent_qzv_key]
        else:
            raise ValueError(f"Unknown latent mode: {mode}")
        self.adata.uns[_ADATA_LATENT_UNS_KEY] = mode
        del self.adata.raw
        all_zeros = csr_matrix(self.adata.X.shape)
        self.adata.X = all_zeros.copy()
        self.adata.layers = {layer: all_zeros.copy() for layer in self.adata.layers}

    @staticmethod
    def _get_latent_fields(mode: LatentDataType):
        if mode == "dist":
            latent_fields = [
                ObsmField(
                    REGISTRY_KEYS.LATENT_QZM_KEY,
                    _SCVI_LATENT_QZM,
                ),
                ObsmField(
                    REGISTRY_KEYS.LATENT_QZV_KEY,
                    _SCVI_LATENT_QZV,
                ),
            ]
        else:
            raise ValueError(f"Unknown latent mode: {mode}")
        latent_fields.append(
            StringUnsField(
                REGISTRY_KEYS.LATENT_MODE_KEY,
                _ADATA_LATENT_UNS_KEY,
            ),
        )
        return latent_fields

    def to_latent_mode(
        self,
        mode: LatentDataType = "dist",
        use_latent_qzm_key: str = "X_latent_qzm",
        use_latent_qzv_key: str = "X_latent_qzv",
    ):
        """
        Put the model into latent mode.

        The model is put into latent mode by registering new anndata fields
        required for latent mode support - latent qzm, latent qzv, and adata uns
        containing latent mode type - and marking the module as latent. Note that
        this modifies the anndata (and subsequently the model and module properties)
        in place. Please make a copy of those objects (before calling this function)
        if needed.

        Parameters
        ----------
        mode
            The latent data type used
        use_latent_qzm_key
            Key to use in `adata.obsm` where the latent qzm params are stored
        use_latent_qzv_key
            Key to use in `adata.obsm` where the latent qzv params are stored
        """
        self._get_latent_adata_from_adata(mode, use_latent_qzm_key, use_latent_qzv_key)
        self.adata_manager.register_new_fields(self.__class__._get_latent_fields(mode))
        self.module.latent_data_type = mode
