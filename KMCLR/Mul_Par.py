import argparse


def parse_args():
	parser = argparse.ArgumentParser(description='Model Params')

	parser.add_argument('--hidden_dim', default=32, type=int)
	parser.add_argument('--gnn_layer', default="[16,16,16]", type=str)
	parser.add_argument('--dataset', default='Tmall', type=str)
	parser.add_argument('--sampNum', default=40, type=int)
	parser.add_argument('--head_num', default=4, type=int)
	parser.add_argument('--lr', default=3e-4, type=float)
	parser.add_argument('--opt_base_lr', default=1e-3, type=float)
	parser.add_argument('--opt_max_lr', default=5e-3, type=float)
	parser.add_argument('--opt_weight_decay', default=1e-4, type=float)
	parser.add_argument('--batch', default=8192, type=int)
	parser.add_argument('--SSL_batch', default=18, type=int)
	parser.add_argument('--reg', default=1e-3, type=float)
	parser.add_argument('--beta', default=0.005, type=float)
	parser.add_argument('--epoch', default=1000, type=int)
	parser.add_argument('--shoot', default=10, type=int)
	parser.add_argument('--inner_product_mult', default=1, type=float)
	parser.add_argument('--drop_rate', default=0.8, type=float)
	parser.add_argument('--slope', type=float, default=0.1)
	parser.add_argument('--patience', type=int, default=100)
	parser.add_argument('--path', default='./datasets/', type=str)
	parser.add_argument('--target', default='buy', type=str)
	parser.add_argument('--isJustTest', default=False , type=bool)


	return parser.parse_args()


args = parse_args()


