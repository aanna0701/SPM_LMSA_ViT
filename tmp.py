    if args.model == 'vit':
        from models.vit_pytorch.vit import ViT     
        model = ViT(img_size=args.input_size, patch_size = 16, num_classes=args.nb_classes, dim=192, 
                    mlp_dim_ratio=2, depth=12, heads=3, dim_head=192//3, pe_dim=64,
                    dropout=args.drop, stochastic_depth=args.drop_path, is_base=args.is_base, eps=0, merging_size=4,
                    no_init=True, is_coord=args.is_coord, is_LSA=args.is_LSA)
