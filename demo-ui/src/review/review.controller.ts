import { Controller, Get, NotFoundException, Res, ServiceUnavailableException } from "@nestjs/common";
import type { Response } from "express";
import { DATA_MODE, DATA_VIA, NoLiveData, ReviewService, Served, UnreadableLiveData } from "./review.service";

// The payload stays the bare schema object; provenance travels in headers:
// X-Data-Source live|sample, X-Data-Mode, X-Data-Via api|disk, and for live
// data X-Data-Updated (ISO time) and X-Data-Stale when monitoring output is old.
async function send<T>(res: Response, load: () => Promise<Served<T>>): Promise<T> {
  res.setHeader("X-Data-Mode", DATA_MODE);
  res.setHeader("X-Data-Via", DATA_VIA);
  res.setHeader("Cache-Control", "no-store");
  let served: Served<T>;
  try {
    served = await load();
  } catch (err) {
    if (err instanceof NoLiveData) throw new NotFoundException(err.message);
    if (err instanceof UnreadableLiveData) throw new ServiceUnavailableException(err.message);
    throw err;
  }
  res.setHeader("X-Data-Source", served.source);
  if (served.updated) res.setHeader("X-Data-Updated", served.updated.toISOString());
  if (served.stale) res.setHeader("X-Data-Stale", "1");
  return served.data;
}

@Controller("api")
export class ReviewController {
  constructor(private readonly reviewService: ReviewService) {}

  @Get("pr-review")
  getPrReview(@Res({ passthrough: true }) res: Response) {
    return send(res, () => this.reviewService.getPrReview());
  }

  @Get("drift")
  getDriftReport(@Res({ passthrough: true }) res: Response) {
    return send(res, () => this.reviewService.getDriftReport());
  }

  @Get("root-cause")
  getRootCause(@Res({ passthrough: true }) res: Response) {
    return send(res, () => this.reviewService.getRootCause());
  }
}
