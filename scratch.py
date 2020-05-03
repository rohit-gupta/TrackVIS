#
# print(segmodel)
# print(tracking_head)

sample = random.randint(0,1790)
sample = 16
# for i in range(sample, sample+1):
for i in range(1790):
    # itr = 11
    print("_" * 120)
    print(" " * 120)
    print("Video number", i)
    print("_" * 120)
    print(" " * 120)
    batch = []
    for j in range(3):
        # vid_imgs, (vid_semantic_targets, vid_centers, vid_offsets, vid_tracking_masks, vid_tracking_ids) = dataset[i+j]
        x = dataset[i + j]
        batch += [x]

    imgs, (semantic_targets, centers, offsets, tracking_masks, tracking_ids) = collate_fn(batch)

    print("Tracking Ids", tracking_ids)

    # img = vid_imgs[0]
    # semantic_target = vid_semantic_targets[0]
    # centers = vid_centers[0]
    # offsets = vid_offsets[0]
    # tracking_masks = vid_tracking_masks[0]
    # tracking_ids = vid_tracking_ids[0]


    # x_offsets = offsets[0]
    # y_offsets = offsets[1]

    # pil_img = ToPILImage()(img)
    # pil_img.show()
    # print("Image Shape", img.shape)
    # print("Center Shape", centers.shape)
    # print("x offset", x_offsets.shape)
    # print("y offset", y_offsets.shape)
    # print("Number of masks", len(tracking_masks), "Number of IDs",len(tracking_ids))
    # print("mask shape", tracking_masks[0].shape, "ID0:", tracking_ids[0])

    # print(np.array(imgs[0]).shape)
    #
    # plt.imshow(centers[0].numpy(), interpolation='none')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(x_offsets.numpy(), interpolation='none')
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(y_offsets.numpy(), interpolation='none')
    # plt.colorbar()
    # plt.show()
    #
    # for mask_i in tracking_masks:
    #     plt.imshow(mask_i.numpy(), interpolation='none')
    #     plt.colorbar()
    #     plt.show()

    # batch_tensor = img.unsqueeze(0).repeat(2, 1, 1, 1)
    print("Batch Shape", imgs.shape)
    segresult = segmodel(imgs)
    print("Semantic result shape", segresult["semantic"].shape)
    print("ASPP shape", segresult["semantic_aspp"].shape)
    print("Instance Center shape", segresult["instance_center"].shape)
    print("Instance Regression shape", segresult["instance_regression"].shape)

    # tracking_masks = [torch.from_numpy(mask) for mask in tracking_masks]

    track_features, pred_class, gt_ids = tracking_head(segresult["semantic_aspp"], tracking_masks, tracking_ids)
    print("Track Features Shape", track_features.shape)
    print("Pred Instance Class Shape", pred_class.shape)
    print("GT IDs", gt_ids)



# imgs[0].show()
# # Image.fromarray(centers, mode="L").save("centers_mask.png")
# # Image.fromarray(x_offsets, mode="L").save("x_mask.png")
# # Image.fromarray(y_offsets, mode="L").save("y_mask.png")
#
# plt.imshow(x_offsets, interpolation='none')
# plt.colorbar()
# plt.show()
#
# plt.imshow(y_offsets, interpolation='none')
# plt.colorbar()
# plt.show()
#
# plt.imshow(centers, interpolation='none')
# plt.colorbar()
# plt.show()
#
#
#
# plt.imshow(semantic_target.sum(axis=-1), interpolation='none')
# plt.colorbar()
# plt.show()
#
# plt.imshow(tracking_masks[0], interpolation='none')
# plt.colorbar()
# plt.show()