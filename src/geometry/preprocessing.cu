#include "../include/MyPolygon.cuh"

void *rasterization_unit(void *args){
	query_context *ctx = (query_context *)args;
	query_context *gctx = ctx->global_ctx;

	std::vector<MyPolygon *> &polygons = *(std::vector<MyPolygon *> *)gctx->target;

	//log("thread %d is started",ctx->thread_id);
	while(ctx->next_batch(10)){
		for(int i=ctx->index;i<ctx->index_end;i++){
			polygons[i]->rasterization(ctx->vpr);
		}
	}
	ctx->merge_global();
	return NULL;
}

void process_rasterization(query_context *gctx){
	log("start rasterizing the referred polygons");
	std::vector<MyPolygon *> &polygons = *(std::vector<MyPolygon *> *)gctx->target;
	assert(polygons.size()>0);
	gctx->index = 0;
	size_t former = gctx->target_num;
	gctx->target_num = polygons.size();

	pthread_t threads[gctx->num_threads];
	query_context ctx[gctx->num_threads];
	for(int i=0;i<gctx->num_threads;i++){
		ctx[i] = *gctx;
		ctx[i].thread_id = i;
		ctx[i].global_ctx = gctx;
	}

	for(int i=0;i<gctx->num_threads;i++){
		pthread_create(&threads[i], NULL, rasterization_unit, (void *)&ctx[i]);
	}

	for(int i = 0; i < gctx->num_threads; i++ ){
		void *status;
		pthread_join(threads[i], &status);
	}

	gctx->index = 0;
	gctx->query_count = 0;
	gctx->target_num = former;
}

void preprocess(query_context *gctx){

	std::vector<MyPolygon *> target_polygons;
	target_polygons.insert(target_polygons.end(), gctx->source_polygons.begin(), gctx->source_polygons.end());
	target_polygons.insert(target_polygons.end(), gctx->target_polygons.begin(), gctx->target_polygons.end());
	gctx->target = (void *)&target_polygons;
	process_rasterization(gctx);
	target_polygons.clear();
	gctx->target = NULL;
}